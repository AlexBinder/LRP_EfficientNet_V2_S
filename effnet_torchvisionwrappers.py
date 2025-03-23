import copy
import math

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchvision

from torch import nn, Tensor

from torchvision.ops import SqueezeExcitation, Conv2dNormActivation

#efficient net code stuff
from torchvision.models._utils import _make_divisible, _ovewrite_named_param #, handle_legacy_interface

from torchvision.models.efficientnet import _MBConvConfig, _efficientnet_conf
from torchvision.models.efficientnet import FusedMBConv,MBConv, MBConvConfig, FusedMBConvConfig
from torchvision.utils import _log_api_usage_once

from basiclrpwrappers import * #mult_wtatosecond, sum_stacked2


class lrpwrapped_squeezeex(SqueezeExcitation):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        #super().__init__(input_channels, squeeze_channels, activation, scale_activation)
        super().__init__(input_channels, squeeze_channels, torch.nn.ReLU, torch.nn.ReLU)
        
        self.multdistribute_wtatosecond = mult_wtatosecond
        

        self.activation = activation
        self.scale_activation = scale_activation
        
        if self.activation is not None:
            self.activation.inplace = False 
        if self.scale_activation is not None:            
            self.scale_activation.inplace = False 


    def forward(self, input: Tensor) -> Tensor:
          scale = self._scale(input)
          #z = scale * input
          z = self.multdistribute_wtatosecond.apply( scale , input ) # wrap by wta rule
          return z 

    def copyfrom(self,sqex):
      
      self.fc1 = copy.deepcopy(sqex.fc1)
      self.fc2 = copy.deepcopy(sqex.fc2)  
      
      self.activation = copy.deepcopy(sqex.activation)
      self.activation.inplace = False 

      self.scale_activation = copy.deepcopy(sqex.scale_activation)
      self.scale_activation.inplace = False 


class MBConv_canon(MBConv):

    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
    
        super().__init__(cnf, stochastic_depth_prob, norm_layer , se_layer )

        self.elt = sum_stacked2() 
        
        #replace selayer!
        blockindex = -1
        
        for ind, bl in enumerate(self.block):
          if not isinstance(bl,Conv2dNormActivation):
            if blockindex<0:
              blockindex = ind
              break
            else:
              print('ERROR: found a not Conv2dNormActivation block at least twice, at inds', blockindex, ind )
              exit()  
        
        if blockindex < 0:
          print('could not find a not Conv2dNormActivation block')
          exit()

        input_channels = self.block[blockindex].fc1.in_channels #nn.Conv2d !
        squeeze_channels = self.block[blockindex].fc1.out_channels #nn.Conv2d !
        activation = copy.deepcopy(self.block[blockindex].activation)
        scale_activation = copy.deepcopy(self.block[blockindex].scale_activation)
        oldfc1 = copy.deepcopy(self.block[blockindex].fc1)
        oldfc2 = copy.deepcopy(self.block[blockindex].fc2)
        
        self.block[blockindex] = lrpwrapped_squeezeex(input_channels , squeeze_channels , activation , scale_activation )
        self.block[blockindex].fc1 = oldfc1 
        self.block[blockindex].fc2 = oldfc2 

    def forward(self, inputs: Tensor) -> Tensor:
        result = self.block(inputs)
        
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            #result += inputs            
            result = self.elt( torch.stack([result,inputs], dim=0) )
            
        return result

class FusedMBConv_canon(FusedMBConv):

    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
    
        super().__init__(cnf, stochastic_depth_prob, norm_layer )
        
        self.elt = sum_stacked2() 

    def forward(self, inputs: Tensor) -> Tensor:
        result = self.block(inputs)
        
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            #result += inputs            
            result = self.elt( torch.stack([result,inputs], dim=0) )
            
        return result



class EfficientNet_forlrpwrap(torch.nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
    ) -> None:
        """
        EfficientNet V1 and V2 main class, for lrpwrapping

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer= nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                #stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                if isinstance(block_cnf, MBConvConfig):
                    blk = MBConv_canon(block_cnf, sd_prob, norm_layer)
                elif isinstance(block_cnf, FusedMBConvConfig):
                    blk = FusedMBConv_canon(block_cnf, sd_prob, norm_layer)                
                else:
                  print('unk class', type(block_cnf))
                  exit()
                stage.append(blk)  
                
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

        for m in self.modules():
          if hasattr(m,'inplace'):
            m.inplace = False

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientnet_forlrpwrap(
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int],
    weights,
    progress: bool,
    **kwargs: Any,
) -> EfficientNet_forlrpwrap:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = EfficientNet_forlrpwrap(inverted_residual_setting, dropout, last_channel=last_channel, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


def efficientnet_v2_s_forlrpwrap(weights: Optional[torchvision.models.EfficientNet_V2_S_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet_forlrpwrap:
    """
    Constructs an EfficientNetV2-S architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_S_Weights
        :members:
    """
    
    weights = torchvision.models.EfficientNet_V2_S_Weights.verify(weights)
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    
    return _efficientnet_forlrpwrap(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.2),
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )


#  Conv2dNormActivation is sequential of conv2d, norm (batchnorm), activation
# todo: check that norm is batchnorm2d, check that activation is inplace=False and not true

def bnafterconv_overwrite_intoconv(conv, bn):

    assert (isinstance(bn,nn.BatchNorm2d))
    assert (isinstance(conv,nn.Conv2d))

    s = torch.sqrt(bn.running_var+bn.eps)
    w = bn.weight.data
    b = bn.bias.data
    m = bn.running_mean.data
    conv.weight = torch.nn.Parameter(conv.weight.data * (w / s).reshape(-1, 1, 1, 1))

    if conv.bias is None:
      conv.bias = torch.nn.Parameter( ((-m) * (w / s) + b).to(conv.weight.dtype) )
    else:
      conv.bias = torch.nn.Parameter(( conv.bias - m) * (w / s) + b)

    return conv



def setbyname2(model,name,value):

    def iteratset(obj,components,value):

      if not hasattr(obj,components[0]):
        return False
      elif len(components)==1:
        setattr(obj,components[0],value)
        #print('found!!', components[0])
        #exit()
        return True
      else:
        nextobj=getattr(obj,components[0])
        return iteratset(nextobj,components[1:],value)

    components=name.split('.')
    success=iteratset(model,components,value)
    return success

def copyfromefficientnet( net, lrp_params, lrp_layer2method):
      
  assert( isinstance(net, torchvision.models.EfficientNet))


  model = efficientnet_v2_s_forlrpwrap()

  updated_layers_names=[]

  last_target_module_name=None
  last_target_module=None
  lrpsqueezeex_name = None


  for src_module_name, src_module in net.named_modules():
  
    print('at src_module_name', src_module_name )
    print(type(src_module))
    foundsth=False
    
    if isinstance(src_module, torchvision.ops.misc.SqueezeExcitation):
      lrpsqueezeex_name  = src_module_name
      print('found an instance of lrpwrapped_squeezeex:',src_module_name)
       
      nm = src_module_name+'.fc1'
      srclr = copy.deepcopy( getattr( src_module,'fc1'  ))
      if False== setbyname2(model, nm , srclr ):
        raise Modulenotfounderror("could not find module "+nm+ " in target net to copy" )
      updated_layers_names.append(srclr)
      
      nm = src_module_name+'.fc2'
      srclr = copy.deepcopy( getattr( src_module,'fc2'  ))
      if False== setbyname2(model, nm , srclr ):
        raise Modulenotfounderror("could not find module "+nm+ " in target net to copy" )
      updated_layers_names.append(srclr)   
      
      nm = src_module_name+'.activation'
      srclr = copy.deepcopy( getattr( src_module,'activation'  ))
      srclr.inplace= False
      if False== setbyname2(model, nm , srclr ):
        raise Modulenotfounderror("could not find module "+nm+ " in target net to copy" )
      updated_layers_names.append(srclr)   
            
      nm = src_module_name+'.scale_activation'
      srclr = copy.deepcopy( getattr( src_module,'scale_activation'  ))
      srclr.inplace= False
      if False== setbyname2(model, nm , srclr ):
        raise Modulenotfounderror("could not find module "+nm+ " in target net to copy" )
      updated_layers_names.append(srclr)          
     
    if isinstance(src_module, nn.modules.linear.Linear):
      foundsth=True
      wrapped = get_lrpwrapperformodule_effnet( copy.deepcopy(src_module) , lrp_params, lrp_layer2method)
      if False== setbyname2(model, src_module_name, wrapped ):
        raise Modulenotfounderror("could not find module "+src_module_name+ " in target net to copy" )
      updated_layers_names.append(src_module_name)

    if isinstance(src_module, nn.modules.conv.Conv2d):
      #store conv2d layers
      foundsth=True
      last_src_module_name=src_module_name
      last_src_module=src_module
      
    #if lrpsqueezeex_name:
    #  if (lrpsqueezeex_name in src_module_name) and (lrpsqueezeex_name != src_module_name):
    #    # !!! single conv layers in squeeze ex dont need to be wrapped for wtatosecond, lol :) 
    #    print('inside lrp squeeze ex:', src_module_name, 'for:', lrpsqueezeex_name)

    if isinstance(src_module, nn.modules.batchnorm.BatchNorm2d):
      # conv-bn chain
      foundsth=True

      specialbn = False


      m = copy.deepcopy(last_src_module)

      #if m.padding[0] >0:
      #  special=True
      
      if False == specialbn:      
        m = bnafterconv_overwrite_intoconv(m , bn = src_module)
        
      wrapped = get_lrpwrapperformodule_effnet( m , lrp_params, lrp_layer2method)

      if False== setbyname2(model, last_src_module_name, wrapped  ):
        raise Modulenotfounderror("could not find module "+nametofind+ " in target net to copy" )
    
      updated_layers_names.append(last_src_module_name)
      
      # wrap batchnorm  
      if False == specialbn:      
        mod2 = resetbn(src_module) 
      else:
        mod2 = copy.deepcopy(src_module) 
      wrapped = get_lrpwrapperformodule_effnet( mod2 , lrp_params, lrp_layer2method)

      if False== setbyname2(model, src_module_name, wrapped ):
        raise Modulenotfounderror("could not find module "+src_module_name+ " in target net to copy" )            
      updated_layers_names.append(src_module_name)
    # end of if


    #if False== foundsth:
    #  print('!untreated layer:',)
    print('\n')
  
  # sum_stacked2 is present only in the targetclass, so must iterate here
  for target_module_name, target_module in model.named_modules():

    if isinstance(target_module, nn.modules.pooling.AdaptiveAvgPool2d):
      wrapped = get_lrpwrapperformodule_effnet( target_module , lrp_params, lrp_layer2method)

      if False== setbyname2(model,target_module_name, wrapped ):
        raise Modulenotfounderror("could not find module "+src_module_name+ " in target net to copy" )            
      updated_layers_names.append(target_module_name)

    if isinstance(target_module, nn.modules.activation.SiLU):
      wrapped = get_lrpwrapperformodule_effnet( target_module , lrp_params, lrp_layer2method)

      if False== setbyname2(model, target_module_name, wrapped ):
        raise Modulenotfounderror("could not find module "+src_module_name+ " in target net to copy" )            
      updated_layers_names.append(target_module_name)
      
    # sum_stacked2
    if isinstance(target_module, sum_stacked2 ):
      wrapped =  get_lrpwrapperformodule_effnet( target_module , lrp_params, lrp_layer2method)
      if False== setbyname2(model, target_module_name, wrapped ):
        raise Modulenotfounderror("could not find module "+target_module_name+ " in target net , impossible!" )            
      updated_layers_names.append(target_module_name)
      
    # se mult
    if isinstance(target_module, mult_wtatosecond):
      # do nothing as it is already wrapped
      updated_layers_names.append(target_module_name)

  print('-------\n\n')
  for target_module_name, target_module in model.named_modules():
    if target_module_name not in updated_layers_names:
      print('not updated:', target_module_name, type(target_module))
  
  return model

def get_lrpwrapperformodule_effnet(module, lrp_params, lrp_layer2method):

  if isinstance(module, nn.modules.activation.ReLU):

    key='nn.ReLU'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)
      
    print('wrap relu')  
      
    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]
    return zeroparam_wrapper_class( module , autogradfunction= autogradfunction )

  elif isinstance(module, nn.modules.batchnorm.BatchNorm2d):

    key='nn.BatchNorm2d'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    print('wrap BatchNorm2d')     
    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]
    return zeroparam_wrapper_class( module , autogradfunction= autogradfunction )

  elif isinstance(module, nn.modules.linear.Linear):

    key='nn.Linear'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    print('wrap linear')     
    #default linearlayer_eps_wrapper_fct()
    autogradfunction = lrp_layer2method[key]
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['linear_eps'] )

  elif isinstance(module, nn.modules.conv.Conv2d): 

      key='nn.Conv2d'
      if key not in lrp_layer2method:
        print("found no dictionary entry in lrp_layer2method for this module name:", key)
        raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

      print('wrap conv2d')     
      autogradfunction = lrp_layer2method[key]
           
      if autogradfunction.__name__ == 'conv2d_beta0_wrapper_fct': # dont want test for derived classes but equality
        return oneparam_wrapper_class(module, autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'] )
      elif autogradfunction.__name__ == 'conv2d_betaany_wrapper_fct': 
        return twoparam_wrapper_class(module, autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'], parameter2 = torch.tensor( lrp_params['conv2d_beta']) )
      elif autogradfunction.__name__ == 'conv2d_betaadaptive_wrapper_fct': 
        return twoparam_wrapper_class(module, autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'], parameter2 = torch.tensor( lrp_params['conv2d_maxbeta']) )  
      else:
        print(autogradfunction, autogradfunction.__name__)
        print('unknown autogradfunction', autogradfunction , key, autogradfunction.__name__ )
        exit()

  elif isinstance(module, nn.modules.pooling.AdaptiveAvgPool2d):

    key='nn.AdaptiveAvgPool2d'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    print('wrap adaptiveavgpool2d')
    #default adaptiveavgpool2d_wrapper_fct()
    autogradfunction = lrp_layer2method[key]
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['pooling_eps'] )

  elif isinstance(module, sum_stacked2): # resnet specific

    key='sum_stacked2'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    print('wrap sum_stacked2')
    #default eltwisesum_stacked2_eps_wrapper_fct()
    autogradfunction = lrp_layer2method[key]
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['eltwise_eps'] )
  
  elif isinstance(module, nn.modules.activation.SiLU ):
    #return zeroparam_wrapper_class( module , relu_wrapper_fct() )

    key='nn.SiLU'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    print('wrap nn.SiLU')
    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]
    return zeroparam_wrapper_class( module , autogradfunction= autogradfunction )
  else:
    print("found no lookup for this module:", module, type(module), type(module).__name__  )
    raise lrplookupnotfounderror( "found no lookup for this module:", module)




