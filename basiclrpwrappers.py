import copy
import torch

class mult_wtatosecond(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x1,x2):
        return x1*x2

    @staticmethod
    def backward(ctx,grad_output):
        return torch.zeros_like(grad_output), grad_output

class sum_stacked2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x): # from X=torch.stack([X0, X1], dim=0)
        assert( x.shape[0]==2 )    
        return torch.sum(x,dim=0)
        
def resetbn(bn):

  assert (isinstance(bn, torch.nn.BatchNorm2d))


  bnc=copy.deepcopy(bn)
  bnc.reset_parameters()

  return bnc

class zeroparam_wrapper_class(torch.nn.Module):
  def __init__(self, module, autogradfunction):
    super().__init__()
    self.module=module
    self.wrapper=autogradfunction

  def forward(self,x):
    y=self.wrapper.apply( x, self.module)
    return y

class oneparam_wrapper_class(torch.nn.Module):
  def __init__(self, module, autogradfunction, parameter1):
    super().__init__()
    self.module=module
    self.wrapper=autogradfunction
    self.parameter1=parameter1

  def forward(self,x):
    y=self.wrapper.apply( x, self.module,self.parameter1)
    return y

class twoparam_wrapper_class(torch.nn.Module):
  def __init__(self, module, autogradfunction, parameter1, parameter2):
    super().__init__()
    self.module=module
    self.wrapper=autogradfunction
    self.parameter1=parameter1
    self.parameter2 = parameter2

  def forward(self,x):
    y=self.wrapper.apply( x, self.module,self.parameter1, self.parameter2)
    return y


class relu_wrapper_fct(torch.autograd.Function): # to be used with generic_activation_pool_wrapper_class(module,this)
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module):
        def configvalues_totensorlist(module):
            propertynames=[]
            values=[]
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################
        #stash module config params and trainable params
        #propertynames,values=configvalues_totensorlist(conv2dclass)
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)
        #input_, conv2dweight, conv2dbias, *values  = ctx.saved_tensors
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=[]
            paramsdict={}
            return paramsdict
        #######################################################################
        #paramsdict=tensorlist_todict(values)
        return grad_output,None


#lineareps_wrapper_fct
class linearlayer_eps_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, eps):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module):
            #[module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']}

            propertynames=['in_features','out_features']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= module.weight.device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device)     
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module)
        epstensor= torch.tensor([eps], dtype=torch.float32, device= x.device) 

        if module.bias is None:
          bias=None
        else:
          bias= module.bias.data.clone()
        ctx.save_for_backward(x, module.weight.data.clone(), bias, epstensor, *values ) # *values unpacks the list

        #print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()

        #print('linear custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, weight, bias, epstensor, *values  = ctx.saved_tensors
        #print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['in_features','out_features']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        if bias is None:
          module=torch.nn.Linear( **paramsdict, bias=False )
        else:
          module=torch.nn.Linear( **paramsdict, bias=True )
          module.bias= torch.nn.Parameter(bias)
                
        #print('conv2dconstr')
        module.weight= torch.nn.Parameter(weight)

        #print('linaer custom input_.shape', input_.shape )
        eps=epstensor.item()
        X = input_.clone().detach().requires_grad_(True)
        R= lrp_backward(_input= X , layer = module , relevance_output = grad_output, eps0 = eps, eps=eps)

        #print('linaer custom R', R.shape )
        #exit()
        return R,None,None


class eltwisesum_stacked2_eps_wrapper_fct(torch.autograd.Function): # to be used with generic_activation_pool_wrapper_class(module,this)
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, stackedx, module,eps):
        def configvalues_totensorlist(module):
            propertynames=[]
            values=[]
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################
        #stash module config params and trainable params
        #propertynames,values=configvalues_totensorlist(conv2dclass)

        epstensor= torch.tensor([eps], dtype=torch.float32, device= stackedx.device) 
        ctx.save_for_backward(stackedx, epstensor )
        return module.forward(stackedx)

    @staticmethod
    def backward(ctx, grad_output):
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)
        stackedx,epstensor  = ctx.saved_tensors
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=[]
            paramsdict={}
            return paramsdict
        #######################################################################
        #paramsdict=tensorlist_todict(values)

        #X0 = x1.clone().detach() #.requires_grad_(True)
        #X1 = x2.clone().detach() #.requires_grad_(True)
        #X=torch.stack([X0, X1], dim=0) # along a new dimension!

        X = stackedx.clone().detach().requires_grad_(True)

        eps=epstensor.item()

        s2=sum_stacked2().to(X.device)
        Rtmp= lrp_backward(_input= X , layer = s2 , relevance_output = grad_output, eps0 = eps, eps=eps)

        #R0=Rtmp[0,:]
        #R1=Rtmp[1,:]

        return Rtmp,None,None

class adaptiveavgpool2d_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, eps):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module,device):

            propertynames=['output_size']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= device)     
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module,x.device)
        epstensor = torch.tensor([eps], dtype=torch.float32, device= x.device) 
        ctx.save_for_backward(x, epstensor, *values ) # *values unpacks the list

        #print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()
        #print('adaptiveavg2d custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, epstensor, *values  = ctx.saved_tensors
        #print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['output_size']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)
        eps=epstensor.item()

        #class instantiation
        layerclass= torch.nn.AdaptiveAvgPool2d(**paramsdict)

        #print('adaptiveavg2d custom input_.shape', input_.shape )

        X = input_.clone().detach().requires_grad_(True)
        R= lrp_backward(_input= X , layer = layerclass , relevance_output = grad_output, eps0 = eps, eps=eps)

        #print('adaptiveavg2dcustom R', R.shape )
        #exit()
        return R,None,None

class conv2d_beta0_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, lrpignorebias):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module):
            #[module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']}

            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= module.weight.device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device)     
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################

        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module)

        if module.bias is None:
          bias=None
        else:
          bias= module.bias.data.clone()
        lrpignorebiastensor=torch.tensor([lrpignorebias], dtype=torch.bool, device= module.weight.device)
        ctx.save_for_backward(x, module.weight.data.clone(), bias, lrpignorebiastensor, *values ) # *values unpacks the list

        #print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()

        #print('conv2d custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, conv2dweight, conv2dbias, lrpignorebiastensor, *values  = ctx.saved_tensors
        #print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        if conv2dbias is None:
          module= torch.nn.Conv2d( **paramsdict, bias=False )
        else:
          module= torch.nn.Conv2d( **paramsdict, bias=True )
          module.bias= torch.nn.Parameter(conv2dbias)
                
        #print('conv2dconstr')
        module.weight= torch.nn.Parameter(conv2dweight)

        #print('conv2dconstr weights')

        pnconv = posnegconv(module, ignorebias = lrpignorebiastensor.item())


        #print('conv2d custom input_.shape', input_.shape )

        X = input_.clone().detach().requires_grad_(True)
        R= lrp_backward(_input= X , layer = pnconv , relevance_output = grad_output, eps0 = 1e-12, eps=0)
        #R= lrp_backward(_input= X , layer = pnconv , relevance_output = torch.ones_like(grad_output), eps0 = 1e-12, eps=0)
        #print( 'beta 0 negR' ,torch.mean((R<0).float()).item() ) # no neg relevance

        #print('conv2d custom R', R.shape )
        #exit()
        return R,None, None

class posnegconv(torch.nn.Module):

    def _clone_module(self, module):
        clone = torch.nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                     **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
      super(posnegconv, self).__init__()

      self.posconv=self._clone_module(conv)
      self.posconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(min=0) ).to(conv.weight.device)

      self.negconv=self._clone_module(conv)
      self.negconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(max=0) ).to(conv.weight.device)

      #ignbias=True
      #ignorebias=False
      if ignorebias==True:
        self.posconv.bias=None
        self.negconv.bias=None
      else:
          if conv.bias is not None:
              self.posconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(min=0) )
              self.negconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(max=0) )

      #print('done init')

    def forward(self,x):
        vp= self.posconv ( torch.clamp(x,min=0)  )
        vn= self.negconv ( torch.clamp(x,max=0)  )
        return vp+vn

class invertedposnegconv(torch.nn.Module):

    def _clone_module(self, module):
        clone = torch.nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                     **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
      super(invertedposnegconv, self).__init__()

      self.posconv=self._clone_module(conv)
      self.posconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(min=0) ).to(conv.weight.device)

      self.negconv=self._clone_module(conv)
      self.negconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(max=0) ).to(conv.weight.device)

      self.posconv.bias=None
      self.negconv.bias=None
      if ignorebias==False:
          if conv.bias is not None:
              self.posconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(min=0) )
              self.negconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(max=0) )

      #print('done init')

    def forward(self,x):
        #vp= self.posconv ( torch.clamp(x,min=0)  )
        #vn= self.negconv ( torch.clamp(x,max=0)  )
        
        vp= self.posconv (  torch.clamp(x,max=0)  ) #on negatives
        vn= self.negconv ( torch.clamp(x,min=0) ) #on positives
        #return vn
        #return vp
        #print( 'negfwd pos?' ,torch.mean((vp>0).float()).item() , torch.mean((vn>0).float()).item() )
        
        return vp+vn # zero or neg


class conv2d_betaany_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, lrpignorebias, beta):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module):
            #[module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']}

            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= module.weight.device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device)     
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module)

        if module.bias is None:
          bias=None
        else:
          bias= module.bias.data.clone()
        lrpignorebiastensor=torch.tensor([lrpignorebias], dtype=torch.bool, device= module.weight.device)
        ctx.save_for_backward(x, module.weight.data.clone(), bias, lrpignorebiastensor, beta, *values ) # *values unpacks the list

        #print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()

        #print('conv2d custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, conv2dweight, conv2dbias, lrpignorebiastensor, beta, *values  = ctx.saved_tensors
        #print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        if conv2dbias is None:
          module= torch.nn.Conv2d( **paramsdict, bias=False )
        else:
          module= torch.nn.Conv2d( **paramsdict, bias=True )
          module.bias= torch.nn.Parameter(conv2dbias)
                
        #print('conv2dconstr')
        module.weight= torch.nn.Parameter(conv2dweight)

        #print('conv2dconstr weights')

        pnconv = posnegconv(module, ignorebias = lrpignorebiastensor.item())
        invertedpnconv = invertedposnegconv(module, ignorebias = lrpignorebiastensor.item())


        #print('conv2d custom input_.shape', input_.shape )

        X = input_.clone().detach().requires_grad_(True)
        R1= lrp_backward(_input= X , layer = pnconv , relevance_output = grad_output, eps0 = 1e-12, eps=0)
        R2= lrp_backward(_input= X , layer = invertedpnconv , relevance_output = grad_output, eps0 = -1e-12, eps=0)
        #R2= lrp_backward(_input= X , layer = invertedpnconv , relevance_output = torch.ones_like(grad_output), eps0 = -1e-12, eps=0)
        #print(beta.item(), 'negR, posR' ,torch.mean((R2<0).float()).item(), torch.mean((R2>0).float()).item()  ) #only pos or 0

        R = (1+beta)*R1-beta*R2
        return R, None, None, None

class conv2d_betaadaptive_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, lrpignorebias, maxbeta):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module):
            #[module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']}

            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= module.weight.device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device)     
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module)

        if module.bias is None:
          bias=None
        else:
          bias= module.bias.data.clone()
        lrpignorebiastensor=torch.tensor([lrpignorebias], dtype=torch.bool, device= module.weight.device)
        ctx.save_for_backward(x, module.weight.data.clone(), bias, lrpignorebiastensor, maxbeta, *values ) # *values unpacks the list

        #print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()

        #print('conv2d custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, conv2dweight, conv2dbias, lrpignorebiastensor, maxbeta, *values  = ctx.saved_tensors
        #print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        if conv2dbias is None:
          module= torch.nn.Conv2d( **paramsdict, bias=False )
        else:
          module= torch.nn.Conv2d( **paramsdict, bias=True )
          module.bias= torch.nn.Parameter(conv2dbias)
                
        #print('conv2dconstr')
        module.weight= torch.nn.Parameter(conv2dweight)

        #print('conv2dconstr weights')

        pnconv = posnegconv(module, ignorebias = lrpignorebiastensor.item())
        invertedpnconv = invertedposnegconv(module, ignorebias = lrpignorebiastensor.item())


        #print('get ratios per output')
        # betatensor =  -neg / conv but care for zeros
        X = input_.clone().detach()
        out = module(X)
        negscores = -invertedpnconv(X)
        betatensor = torch.zeros_like(out)
        #print('out.device',out.device, negscores.device)
        betatensor[ out>0 ] = torch.minimum( negscores[ out>0 ] / out [ out>0 ], maxbeta.to(out.device))
        #betatensor = torch.mean( betatensor ,dim=1, keepdim=True )
        
        
        #print('conv2d custom input_.shape', input_.shape )

        X.requires_grad_(True)
        R1= lrp_backward(_input= X , layer = pnconv , relevance_output = grad_output * (1+betatensor), eps0 = 1e-12, eps=0)
        R2= lrp_backward(_input= X , layer = invertedpnconv , relevance_output = grad_output * betatensor, eps0 = 1e-12, eps=0)

        #print('betatensor',betatensor.shape, R1.shape, out.shape, X.shape)

        R = R1 -R2
        return R, None, None, None

#######################################################
#######################################################
# #base routines
#######################################################
#######################################################

def safe_divide(numerator, divisor, eps0,eps):
    return numerator / (divisor + eps0 * (divisor == 0).to(divisor) + eps * divisor.sign() )

def lrp_backward(_input, layer, relevance_output, eps0, eps):
    """
    Performs the LRP backward pass, implemented as standard forward and backward passes.
    """
    #if hasattr(_input,'grad'):
    if _input.grad is not None:
      _input.grad.zero_()
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(_input)
    S = safe_divide(relevance_output_data, Z.clone().detach(), eps0, eps)

    #print('started backward')
    Z.backward(S)
    #print('finished backward')
    relevance_input = _input.data * _input.grad.data
    return relevance_input #.detach()

        
