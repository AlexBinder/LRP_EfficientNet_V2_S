

import numpy as np
import matplotlib.pyplot as plt
from getimagenetclasses import *
from dataset_imagenet2500 import dataset_imagenetvalpart_nolabels

import torch
import torchvision

from basiclrpwrappers import *
from effnet_torchvisionwrappers import copyfromefficientnet


def getmodel_imagenet():

  model = torchvision.models.efficientnet_v2_s(weights= torchvision.models.EfficientNet_V2_S_Weights.DEFAULT )  
  model.eval()
  
  return model  


def ftmaphook2(module, input, output, dic, nam):
  dic[nam] = copy.deepcopy(output.clone())
  #print('at', nam)
  return

def ftmaphook(module, input, output) -> None:

  #print( 'module INPUT feature map size: ',  input[0].shape)
  #print(module)
  print( 'module OUTPUT feature map size: ',  output.shape,'\n--------------\n\n')


def cmpforwardpass2():

  torch.manual_seed(6)

  m2 = getmodel_imagenet()


  lrp_params_def1={
    'conv2d_ignorebias': True, 
    'eltwise_eps': 1e-6,
    'linear_eps': 1e-6,
    'pooling_eps': 1e-6,
    'conv2d_maxbeta': 2.0,
  }

  lrp_layer2method={
    'nn.SiLU':          relu_wrapper_fct,
    'nn.BatchNorm2d':   relu_wrapper_fct,
    'nn.Conv2d':        conv2d_beta0_wrapper_fct, #conv2d_betaadaptive_wrapper_fct, #conv2d_beta0_wrapper_fct,
    'nn.Linear':        linearlayer_eps_wrapper_fct,  
    'nn.AdaptiveAvgPool2d': adaptiveavgpool2d_wrapper_fct,
    'sum_stacked2': eltwisesum_stacked2_eps_wrapper_fct,
  }
  
  model = copyfromefficientnet( m2, lrp_params_def1, lrp_layer2method)


  m2.eval()
  model.eval()

  transforms =   torchvision.models.EfficientNet_V2_S_Weights.DEFAULT.transforms()  
  
  

  ft2 = {}     
  allnames = []
  for nm2,mod2 in m2.named_modules():
    if isinstance( mod2, nn.modules.batchnorm.BatchNorm2d):
      mod2.register_forward_hook( partial( ftmaphook2, dic = ft2, nam= nm2 ) )
      allnames.append(nm2)
  
  ft1 = {}
  for nm1,mod1 in model.named_modules():
      if nm1 in allnames:
        mod1.register_forward_hook( partial( ftmaphook2, dic = ft1, nam= nm1 ) )

  im = torch.randint(low=0,high=255,size=(1,3,224,224))  
  
  reps = 1
  alldiffs = []
  for r in range(reps):
    print('at',r)
    im = torch.randint(low=0,high=255,size=(1,3,224,224))
  
    im = transforms(im)
    with torch.no_grad():
      out1 = m2(im)
      out2 = model(im)
    
      diff = torch.norm(out1-out2)
      alldiffs.append(diff)
    print('diff',diff.item())


    for nm in allnames:
        d2 = torch.norm( ft1[nm]-ft2[nm])
        print('d2',nm,d2.item())    
        
        #print(torch.mean( ft1[nm]-ft2[nm]).item() )
        #print(torch.mean( torch.abs( ft1[nm]-ft2[nm])).item() )
        #print(torch.std( ft1[nm]-ft2[nm]).item() )    
        
        #print(ft1[nm]-ft2[nm])
        
        
        print(ft1[nm].shape) 
        
        for c in range(ft1[nm].shape[1]):
          print('at c',c)
          print(torch.mean( torch.abs( ft1[nm][0,c]-ft2[nm][0,c])).item() )    
          print(torch.std( ft1[nm][0,c]-ft2[nm][0,c]).item() )    
            
        break  
    
  alldiffs = torch.Tensor(alldiffs)  
  print(torch.mean(alldiffs), torch.max(alldiffs))

  for nm2,mod2 in m2.named_modules():
    if isinstance( mod2, nn.modules.batchnorm.BatchNorm2d):
      print(mod2)
      print(mod2.running_var, mod2.running_mean)
      break
  for nm2,mod2 in model.named_modules():
    if isinstance( mod2, nn.modules.batchnorm.BatchNorm2d):
      print(mod2)
      print(mod2.running_var, mod2.running_mean)
      break  




def imshow2(hm,imgtensor, title=None, q=100):

    def invert_normalize(ten, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
      print(ten.shape)
      s=torch.tensor(np.asarray(std,dtype=np.float32)).unsqueeze(1).unsqueeze(2)
      m=torch.tensor(np.asarray(mean,dtype=np.float32)).unsqueeze(1).unsqueeze(2)

      res=ten*s+m
      return res

    def showimgfromtensor(inpdata):

      ts=invert_normalize(inpdata)
      a=ts.data.squeeze(0).numpy()
      saveimg=(a*255.0).astype(np.uint8)

      #PIL.Image.fromarray(np.transpose(saveimg,[1,2,0]), 'RGB').show() #.save(savename)
    ######## 


    fig, axs = plt.subplots(1, 2 )

    hm = hm.squeeze().sum(dim=0).numpy()

    clim = np.percentile(np.abs(hm), q)
    hm = hm / clim
    #hm = gregoire_black_firered(hm)
    #axs[1].imshow(hm)
    axs[1].imshow(hm, cmap="seismic", clim=(-1, 1))
    axs[1].axis('off')

    ts=invert_normalize(imgtensor.squeeze())
    a=ts.data.numpy().transpose((1, 2, 0))
    axs[0].imshow(a)
    axs[0].axis('off')
    if title:
      plt.title(title)

    plt.show()



def run1():

  weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
  model = efficientnet_v2_s_forlrpwrap(weights)

  for nm,mod in model.named_modules():
    print(nm,'|wr')
    print(mod)
    print('-------\n')

  '''
  m2 = getmodel_imagenet()
  print('here??')
  for nm,mod in m2.named_modules():
    print(nm,'|||')
    print(mod)
    print('-------\n')
  '''  

def cmpforwardpass1():

  torch.manual_seed(3)

  m2 = getmodel_imagenet()

  weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
  model = efficientnet_v2_s_forlrpwrap(weights)
  
  transforms =   torchvision.models.EfficientNet_V2_S_Weights.DEFAULT.transforms()
  
  reps = 10

  m2.eval()
  model.eval()
  
  alldiffs = []
  for r in range(reps):
    print('at',r)
    im = torch.randint(low=0,high=255,size=(1,3,128,128))
  
    im = transforms(im)
    
    out1 = m2(im)
    out2 = model(im)
    
    diff = torch.norm(out1-out2)
    alldiffs.append(diff)
  
  alldiffs = torch.Tensor(alldiffs)  
  print(torch.mean(alldiffs), torch.max(alldiffs))

def test_model5(dataloader,  model, device):

  model.train(False)

  for data in dataloader:
    # get the inputs
    #inputs, labels, filenames = data
    inputs=data['image']
    labels=data['label']    
    fns=data['filename']  

    inputs = inputs.to(device).clone()
    labels = labels.to(device)

    inputs.requires_grad=True

    print(inputs.requires_grad)
    with torch.enable_grad():
      outputs = model(inputs)

    vals,cls = torch.max(outputs, dim=1)
    outputs[0,cls].backward()

    print(inputs.grad.shape)
    rel=inputs.grad.data
    print( torch.max(rel), torch.mean(rel) )

    clsss=get_classes()


    with torch.no_grad():

      print('shp ', outputs.shape)
      vals,cls = torch.max(outputs, dim=1)
      m=torch.mean(outputs)
      print(  vals.item(), clsss[cls.item()], m.item() )
      print(fns)

    imshow2(rel.to('cpu'),imgtensor = inputs.to('cpu'))
 
def showsomehm(skip):

  #root_dir='/home/binder/entwurf9/data/imagenetvalimg/'
  root_dir = '/home/binder/experiments/2022/062022lrp/imagenetvalsomeimg/'

  m2 = getmodel_imagenet()


  lrp_params_def1={
    'conv2d_ignorebias': True, 
    'eltwise_eps': 1e-3,
    'linear_eps': 1e-3,
    'pooling_eps': 1e-3,
    'conv2d_maxbeta': 2.0,
  }

  lrp_layer2method={
    'nn.SiLU':          relu_wrapper_fct,
    'nn.BatchNorm2d':   relu_wrapper_fct,
    'nn.Conv2d':        conv2d_betaadaptive_wrapper_fct, #conv2d_betaadaptive_wrapper_fct, #conv2d_beta0_wrapper_fct,
    'nn.Linear':        linearlayer_eps_wrapper_fct,  
    'nn.AdaptiveAvgPool2d': adaptiveavgpool2d_wrapper_fct,
    'sum_stacked2': eltwisesum_stacked2_eps_wrapper_fct,
  }
  
  model = copyfromefficientnet( m2, lrp_params_def1, lrp_layer2method)

  m2.eval()
  model.eval()

  transforms2 =   torchvision.models.EfficientNet_V2_S_Weights.DEFAULT.transforms() 

  dset= dataset_imagenetvalpart_nolabels(root_dir, maxnum=1, transform= transforms2, skip= skip)
  dataloader =  torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False) #, num_workers=1) 

  #device = torch.device('cuda:0')
  device = torch.device('cpu')
  model.to(device)

  test_model5(dataloader = dataloader,  model = model, device = device )
 
      
if __name__ =='__main__':
  
  #run1()
  #cmpforwardpass1()
  #cmpforwardpass2()
  
  skip = 81
  # 16,20,24,21 # 68,81,90,30,99,126 (93) # 81 is lady and dog
  showsomehm(skip)
  
