# zernike_transform

## How to start:

- Initialize class
```
Z = Zernike() 
```
- Load image
```
img = Z.load_img("star.jpg")
```
- Crop image to square, x and y - center of img
```
img = Z.crop_image(img=img,
                   x=img.shape[1] // 2,
                   y=img.shape[0] // 2,
                   R=img.shape[0] // 2,
                   show=True)
```
- create zernike polynomials and set depth
```
Z.zernike_polynomials(img, max_depth=40, save=True)
```
- reconstruct image from polynomials 
```
Z.zernike_reconstruct_image(coefs=Z.coefs, R=100, depth=40, save=True)
```


Source image:  
![Source image](star.jpg)  
Output image:  
![Source image](FigDepth40.png)  
