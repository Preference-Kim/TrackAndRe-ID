# Training report

## 2023-10-20

- Image sources: sources=["market1501","msmt17","mydataset"]
    - batch_size_train=128,#*torch.cuda.device_count(),
    - transforms=["random_flip", "random_crop"]
- Learning rate & decay
    - lr=0.00345
    - weight_decay=0.0011
    - max_epoch=320
    - lr_scheduler="multi_step",
    - stepsize=[40, 100, 200]

```bash
Final test                                                                                                                  
##### Evaluating market1501 (source) #####                                                                                     
Extracting features from query set ...                                                                                         
Done, obtained 3368-by-512 matrix                                                                                              
Extracting features from gallery set ...                                                                                       
Done, obtained 15913-by-512 matrix                                                                                             
Speed: 0.0218 sec/batch                                                                                                        
Computing distance matrix with metric=cosine ...                                                                               
Computing CMC and mAP ...                                                                                                      
** Results **                                                                                                                  
mAP: 58.6%                                                                                                                     
CMC curve                                                                                                                      
Rank-1  : 79.9%                                                                                                                
Rank-5  : 91.7%                                                                                                                
Rank-10 : 94.7%                                                                                                                
Rank-20 : 96.9%                                                                                                                
##### Evaluating msmt17 (source) #####                                                                                         
Extracting features from query set ...                                                                                         
Done, obtained 11659-by-512 matrix                                                                                             
Extracting features from gallery set ...                                                                                       
Done, obtained 82161-by-512 matrix                                                                                             
Speed: 0.0207 sec/batch                                                                                                        
Computing distance matrix with metric=cosine ...                                                                               
Computing CMC and mAP ...                                                                                                      
** Results **                                                  
mAP: 22.9%                                                                                                                     
CMC curve                                                                                                                      
Rank-1  : 47.4%                                                                                                                
Rank-5  : 63.7%                                                                                                                
Rank-10 : 70.8%                                                                                                                Rank-20 : 77.2%                                                                                                                
##### Evaluating mydataset (source) #####                                                                                      
Extracting features from query set ...                                                                                         Done, obtained 70-by-512 matrix                                                                                                
Extracting features from gallery set ...                                                                                       
Done, obtained 450-by-512 matrix                                                                                               
Speed: 0.0215 sec/batch                                                                                                        
Computing distance matrix with metric=cosine ...                                                                               
Computing CMC and mAP ...                                                                                                      
** Results **                                                                                                                  
mAP: 88.9%                                                                                                                     
CMC curve                                                                                                                      Rank-1  : 92.6%                                                                                                                
Rank-5  : 94.4%                                                                                                                Rank-10 : 94.4%                                                                                                                
Rank-20 : 94.4%                                                                                                                Checkpoint saved to "log/triplet/images/model/model.pth.tar-320"                                                               
Elapsed 9:43:37
```

## 2023-10-23

### i

- Image sources: sources=["market1501","msmt17","mydataset"]
    - batch_size_train=128,#*torch.cuda.device_count(),
    - transforms=["random_flip", "random_crop"]
- Learning rate & decay
    - lr=0.00345
    - weight_decay=0.0011
    - max_epoch=600
    - lr_scheduler="multi_step",
    - stepsize=[120, 250, 400]

### ii

- Image sources: sources=["market1501","msmt17","mydataset"]
    - batch_size_train=128,#*torch.cuda.device_count(),
    - transforms=["random_flip", "random_crop"]
- Learning rate & decay
    - lr=0.00355
    - weight_decay=0.0011
    - max_epoch=600
    - lr_scheduler="multi_step",
    - stepsize=[120, 200, 300]
