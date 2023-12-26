# CRISPIM-A-U-Net-based-denoising-technique-for-severely-noisy-3D-SPIM-Light-Sheet-Microscope-images

2023-12-26, Wonjin Cho @ KAIST (cerdonis_cho@kaist.ac.kr)


# **1. PROJECT GOAL AND BACKGROUND**

Selective Plane Illumination Microscopy (SPIM), also known as light-sheet microscopy, has become an increasingly important tool in biology [1]. SPIM is a fluorescence optical microscopy technique that employs a planar illumination and an orthogonally oriented detection path [2]. While other common microscopy techniques, such as widefield and confocal microscopy, have inherent disadvantages such as limited sectioning, increased phototoxicity and a slower frame rate, SPIM circumvents these issues by selectively illuminating only the portion of the sample being imaged, enabling high-throughput volumetric imaging of an intact sample with minimal photobleaching [3]. 
An inevitable issue in digital imaging, including SPIM, is the presence of noise [4]. Noise in digital images is a random variation of brightness or color information, and is usually an aspect of electronic noise. It can be produced by the image sensor and circuitry of a scanner or digital camera. Noise is always present in digital images during the image capture, coding, transmission, and processing stages [5].
The noise problem in SPIM is more salient than in other microscopy modalities [6], [7]. This is primarily due to the interaction between the excitation sheet and the detection objective point spread function (PSF) of a SPIM [3]. The spatial variation of the PSF is determined by this interaction, leading to spatially varying blur and a combination of Poisson and Gaussian noise. 
A number of techniques were developed to denoise the microscopy images [8]–[10]. Recent advancements in denoising microscopy images have leveraged various machine learning techniques to improve image quality. For instance, Fuentes-Hurtado et al. proposed a novel framework for few-shot microscopy image denoising that combines generative adversarial networks (GANs) and contrastive learning [9]. Their approach, which was demonstrated on three well-known microscopy imaging datasets, drastically reduces the amount of training data required while retaining the quality of the denoising. Another study by Joucken et al. focused on denoising scanning tunneling microscopy (STM) images [11]. They trained a convolutional neural network on STM images simulated based on a tight-binding electronic structure model. The network was trained on a set of simulated images with varying characteristics such as tip height, sample bias, atomic-scale defects, and non-linear background. Their approach was found to be superior to commonly-used filters in the removal of noise as well as scanning artifacts.
These studies highlight the potential of machine learning techniques in denoising microscopy images, providing a foundation for further research. However, the denoising methods that targets the SPIM images are yet to be explored. Denoising SPIM images might require a different approach compared to the denoising of other 2D microscopy images, because the 3D image of SPIM is a stack of images along the height, the continuity between the adjacent images in a 3D stack must remain intact. If there is a technique that can correct the noisy image of SPIM while taking the information from the adjacent images in a 3D stack into account, it will increase the quality of information that one can acquire from the images acquired from SPIM. Here, I demonstrate such technique, 

termed **CRISPIM (Convolution-based Resolution Improvement on SPIM images)**. 

CRISPIM utilized a U-Net architecture for learning and correcting 3D images acquired through SPIM that are severely noisy (Gaussian noise with standard deviation up to 40).
By effectively reducing severe noise in 3D SPIM images, CRISPIM enhances the quality of the acquired images, thus will improve the reliability and accuracy of subsequent data analysis. With far-reaching implications for various fields in biology, CRISPIM broadens the applicability of SPIM, making it a more robust tool for imaging under challenging conditions.

# **2. METHODS**

## **2.1	MICE**

8 weeks old male Thy1-M mice (JAX#007788) were purchased from The Jackson Laboratory (ME, USA). All mice used for the experiments were 8-16 weeks old. All animal care and handling were performed in accordance with the directives of the Animal Care and Use Committee of KAIST. The mice were housed in standard conditions (maximum of 5 mice per cage) in a 12-hour light/dark cycle with unrestricted access to food and water. Male 8-20 weeks-old mice were used for all experiments.

## **2.2	SPIM IMAGING**

Mouse brain preservation was carried out based on the previously published SHIELD protocol [12]. The resulting sample was optically cleared using optical clearing solution (OCS), and was mounted in the mounting solution. After the mounting solution was cooled and hardened, the mounted sample was incubated overnight within the light-sheet microscope (LifeCanvas Technologies, SmartSPIM v2, serial number: SS20210716V2 (camera: Hamamatsu C14440-20UP; objective lens: LifeCanvas Instruments LCT3.6MTL (custom lens designed based on Thorlabs TL4X-SAP); filter model: Semrock FF03-525/50, FF01-600/52, FF01-680/42)) reservoir filled with OCS for the complete refractive index matching. Image stacks were acquired through SmartSPIM at the lateral resolution of 1.8μm/px, and the axial resolution of 2μm.The size of each image was 1600px in height and 2000px in width. Primary images were acquired using the acquisition software, SmartSPIM SS Acquisition V3 (LifeCanvas, Cambridge, MA). After the acquisition, images were computationally destriped using Pystripe [13]. For Pystripe, LifeCanvas provided a GUI interface which does not modify the underlying open-source software, but provides a useful interface.

## **2.3	DATA**

For this project, 4 image stacks (Number of images: 2888, 2888, 2408, 2862) were chosen as the training data, and one image stack  (Number of images: 2888) was separately chosen as the test data. For the data preprocessing, an image analysis software, ImageJ 1.53q (NIH, Bethesda, Maryland) (Java 1.8.0_172 64-bit) was used. The chosen stacks were downsampled 4 times, into an image size of 400px in height and 500px in width. After then, each training stack data was added with artificial Gaussian noise of standard deviation of 30 (N=2) or 40 (N=2). For the test stack, the Gaussian noise of a random standard deviation in the range of [30, 40] was added in on-the-fly manner, on Google Colaboratory. The image stacks with added noise served as the input data for the model, while the stacks before adding the noise served as the ground-truth data for training. All the images for the training and testing data were cropped into a size of 256px in height and 256px in width before input to the model.

## **2.4	CRISPIM ON 2D IMAGES**

 As the model to denoise the images, U-Net architecture was chosen. U-Net is a type of convolutional neural network that was originally developed for biomedical image segmentation [14]. The architecture of U-Net is characterized by a contracting path to capture context and a symmetric expanding path that enables precise localization. This unique structure allows U-Net to effectively learn and represent features at various scales, making it particularly effective for tasks such as semantic segmentation.
U-Net has been found to be particularly useful for image denoising [15]. The ability of U-Net to capture and represent features at different scales allows it to effectively distinguish between noise and actual image features. Moreover, the expansive path of U-Net helps in preserving the details of the image during the denoising process [16]. This makes U-Net a powerful tool for image denoising, capable of improving the quality of images while preserving their important features.
As a preliminary test on the image denoising through U-Net, a publically available work on the 2D image denoising U-Net model was adopted [17] (https://github.com/jabascal/ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras/blob/main/Train_model_denoising2D.py). The model was built with Keras, while utilizing TensorFlow as the backend.
 ![2d](https://github.com/Cerdonis/CRISPIM-A-U-Net-based-denoising-technique-for-severely-noisy-3D-SPIM-Light-Sheet-Microscope-images/assets/23471213/3f774a4f-99d7-45e5-9455-98c32ab13c7a)

**Figure 1. The architecture of 2D CRISPIM**
 
   The architecture of the model is shown in Figure 1. This model was trained with random selection of noisy and ground-truth image pairs from the 4 image stacks explained above, through 1000 epochs with batch size of 16 pairs (validation data = 1 pair per epoch). The training took 2 hours and 6 minutes, using V100 GPU of Google Colaboratory Pro.

## **2.5	CONVENTIONAL DENOISING METHODS**

To evaluate the performance of CRISPIM on 2D images, two denoising methods commonly utilized through an importable library on Google Colaboratory was chosen.
skimage.restoration.denoise_tv_bregman is a tool that performs total variation denoising using split-Bregman optimization [18]. Parameter of weight=0.1 was used.
cv2.fastNlMeansDenoising is a tool that perform image denoising using Non-local Means Denoising algorithm [19]. Noise of the image processed with this function is expected to be a Gaussian white noise. To use this tool, the images to be denoised were first converted into 8-bit images. Parameters of None, h=10, templateWindowSize=7, searchWindowSize=21 were used.
2.6	CRISPIM ON 3D IMAGE STACKS
Based on the U-Net architecture of 2D CRISPM, an original design of 3D CRISPIM architecture was devised to include additional layers with larger filter size, as well as three skip connections. The model was also built with Keras, while utilizing TensorFlow as the backend. 
 ![3d](https://github.com/Cerdonis/CRISPIM-A-U-Net-based-denoising-technique-for-severely-noisy-3D-SPIM-Light-Sheet-Microscope-images/assets/23471213/4e554de1-3ec9-4119-a887-fd64257630d5)

**Figure 2. The architecture of 3D CRISPIM**

   The architecture of the model is shown in Figure 2. Unlike 2D CRISPIM, 3D CRISPIM takes input of a noisy ‘mini stack’, which is 3D image that consists of 16 adjacent noisy 2D images from the SPIM data, and predicts single 2D image as an output. The mini stack was generated by stacking 8 adjacent images above (lower indices) the target images, the target image itself, and 7 adjacent images below (higher indices). This model was trained with random selection of noisy and ground-truth image pairs from the 4 image stacks explained above, through 8000 epochs with batch size of 4 ‘mini stack – ground truth image’ pairs (validation data = 1 pair per epoch). The combination of T4 and V100 GPU of Google Colaboratory Pro was used for the training.

# **3. RESULTS**

## **3.1	THE PERFORMANCE OF 2D CRISPIM**
After training, the performance of the saved model was evaluated using the test data. As previously explained, random Gaussian noise was added to the test images before being an input to the model. 20 images were randomly selected from the test data stack for the evaluation of the model. As the evaluation metrics to quantify the fidelity of the denoising, PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), and PCC (Pearson Correlation Coefficient) were calculated from the denoised and ground-truth images from each denoising method (Figure 3). The resulting images from each denoising method was visualized (Figure 4).
![image](https://github.com/Cerdonis/CRISPIM-A-U-Net-based-denoising-technique-for-severely-noisy-3D-SPIM-Light-Sheet-Microscope-images/assets/23471213/1a9f7163-12c0-4a08-803b-ccf06d6342e3)

**Figure 3. PSNR, SSIM and PCC comparison between 2D CRISPIM and conventional denoising methods**

   TV: skimage.restoration.denoise_tv_bregman, cv2: cv2.fastNlMeansDenoising, Prediction: 2D CRISPIM. 

![image](https://github.com/Cerdonis/CRISPIM-A-U-Net-based-denoising-technique-for-severely-noisy-3D-SPIM-Light-Sheet-Microscope-images/assets/23471213/42a27707-8be7-4c8d-b346-425cce60abd1)

**Figure 4. The resulting images from each denoising method**

Original: noisy image before denoising, TV: skimage.restoration.denoise_tv_bregman, cv2: cv2.fastNlMeansDenoising, Prediction: 2D CRISPIM. 

|         | PSNR | SSIM  | PCC |
| ------------- | ------------- |------------- | ------------- |
| TV  | 69.6290   | 0.9998   | 0.9718   |
| 2D CRISPIM   | 71.7905   | 0.9999   | 0.9845   |
| CV2   | 60.3469   | 0.9983   | 0.7416   |

**Table 1. Average PSNR, SSIM and PCC comparison between 2D CRISPIM and conventional denoising methods**

   TV: skimage.restoration.denoise_tv_bregman, cv2: cv2.fastNlMeansDenoising.

As the result, I found out that 2D CRISPIM showed much higher performance compared to denoise_tv_bregman and fastNlMeansDenoising, while the gap of the performance much greater in case of fastNlMeansDenoising (Table 1). This result suggests that current denoising approach using a U-Net shows promising prospect, which implies a technique with higher performance when the spatial information of the image to be denoised is taken into account.

## **3.2	THE PERFORMANCE OF 3D CRISPIM**

![image](https://github.com/Cerdonis/CRISPIM-A-U-Net-based-denoising-technique-for-severely-noisy-3D-SPIM-Light-Sheet-Microscope-images/assets/23471213/aec327d7-1913-437b-a214-1e820eee80c2)

**Figure 5. PSNR, SSIM and PCC comparison between 2D CRISPIM and 3D CRISPIM**

![image](https://github.com/Cerdonis/CRISPIM-A-U-Net-based-denoising-technique-for-severely-noisy-3D-SPIM-Light-Sheet-Microscope-images/assets/23471213/5d978013-7c43-4487-abad-9afcff7b12c7)

**Figure 6. The resulting images from 2D CRISPIM and 3D CRISPIM**

|         | PSNR | SSIM  | PCC |
| ------------- | ------------- |------------- | ------------- |
| 2D CRISPIM  | 70.5744   | 0.9999   | 0.9853   |
| 3D CRISPIM   | 64.8643   | 0.9993   | 0.9544   |

**Table 2. Average PSNR, SSIM and PCC comparison between 2D CRISPIM and 3D CRISPIM**

The performance of the 3D CRISPIM model was evaluated using the same test data, and same metrics explained above. 
While the resulting prediction from 3D CRISPIM well captured the details presented in the ground-truth (Figure 6), its performance in PSNR, SSIM and PCC was inferior compared to that of 2D CRISPIM (Table 2). Moreover, the variance of the metrics were higher than 2D CRISPIM (Figure 5).

# **4.	SUMMARY AND DISCUSSION**

In this work, I built two U-Net based image denoising models, 2D and 3D CRISPIM. 2D CRISPIM works on a single image input, while 3D CRISPIM receives a stack of 16 images to consider the information from the adjacent images. 
While both models showed good performance when the resulting prediction and the ground-truth is visually compared with human eye, 2D CRISPIM performed better in context of PSNR, SSIM and PCC over 3D CRISPIM. I can think of some reasons responsible for this outcome opposing my expectations.
The first possible reason is the negative effect from the adjacent images. While the images above and below the Z-coordinate of the target images have a very similar morphology with the target image, it is not exactly the same. The deviation becomes greater as the adjacent image is farther from the target image. Although I first expected that this similar images might give some useful information to the model, the stack of 16 might have been including information that deviates too much from the target image. For the future applications, reducing  the mini-stack height might be a good approach.
The second possible reason is overfitting issue. While the number of parameters in 2D CRISPIM is 332641, 3D CRISPIM has an increasingly higher number of parameters of 5836033. Although a model that works on a 3D image would require more number of parameters, excessive number of parameters may cause an overfitting, causing the performance to be lowered. 
In common, there is a limitation that the size of the image patch that the model works on is small compared to the actual size of raw image tile of SPIM (1600px * 2000px), we need an additional technique to stitch the predictions of the model to make it into an intact tile. 

# **5.	REFERENCES**

[1]	J. Huisken and D. Y. R. Stainier, “Selective plane illumination microscopy techniques in developmental biology,” Development, vol. 136, no. 12, pp. 1963–1975, 2009, doi: 10.1242/dev.022426.

[2]	Z. Lavagnino, F. C. Zanacchi, and A. Diaspro, “Selective Plane Illumination Microscopy (SPIM) BT  - Encyclopedia of Biophysics,” G. C. K. Roberts, Ed. Berlin, Heidelberg: Springer Berlin Heidelberg, 2013, pp. 2307–2308. doi: 10.1007/978-3-642-16712-6_830.

[3]	B. Toader et al., “Image Reconstruction in Light-Sheet Microscopy: Spatially Varying Deconvolution and Mixed Noise,” J. Math. Imaging Vis., vol. 64, no. 9, pp. 968–992, 2022, doi: 10.1007/s10851-022-01100-3.

[4]	L. J. Van Vliet, F. R. Boddeke, D. Sudar, and I. T. Young, “Image Detectors for Digital Image Microscopy,” Digit. Image Anal. Microbes; Imaging, Morphometry, Fluorometry Motil. Tech. Appl., no. January, pp. 1–25, 1998.

[5]	R. A. Al Mudhafar and N. K. El Abbadi, “Noise in Digital Image Processing: A Review Study,” in 2022 3rd Information Technology To Enhance e-learning and Other Application (IT-ELA), 2022, pp. 79–84. doi: 10.1109/IT-ELA57378.2022.10107965.

[6]	K. Keomanee-Dizon, M. Jones, P. Luu, S. E. Fraser, and T. V Truong, “Extended depth-of-field light-sheet microscopy improves imaging of large volumes at high numerical aperture,” Appl. Phys. Lett., vol. 121, no. 16, p. 163701, Oct. 2022, doi: 10.1063/5.0101426.

[7]	P. Escande, P. Weiss, and W. Zhang, “A Variational Model for Multiplicative Structured Noise Removal,” J. Math. Imaging Vis., vol. 57, no. 1, pp. 43–55, 2017, doi: 10.1007/s10851-016-0667-3.

[8]	W. Meiniel, J.-C. Olivo-Marin, and E. D. Angelini, “Denoising of Microscopy Images: A Review of the State-of-the-Art, and a New Sparsity-Based Method,” IEEE Trans. Image Process., vol. 27, no. 8, pp. 3842–3856, 2018, doi: 10.1109/TIP.2018.2819821.

[9]	F. Fuentes-Hurtado, J.-B. Sibarita, and V. Viasnoff, “Generalizable Denoising of Microscopy Images using Generative Adversarial Networks and Contrastive Learning.” 2023.

[10]	L. Fan, F. Zhang, H. Fan, and C. Zhang, “Brief review of image denoising techniques,” Vis. Comput. Ind. Biomed. Art, vol. 2, no. 1, p. 7, 2019, doi: 10.1186/s42492-019-0016-7.

[11]	F. Joucken et al., “Denoising scanning tunneling microscopy images of graphene with supervised machine learning,” Phys. Rev. Mater., vol. 6, no. 12, p. 123802, 2022, doi: 10.1103/PhysRevMaterials.6.123802.

[12]	Y. G. Park et al., “Protection of tissue physicochemical properties using polyfunctional crosslinkers,” Nat. Biotechnol., vol. 37, no. 1, p. 73, Jan. 2019, doi: 10.1038/nbt.4281.

[13]	J. Swaney, “GitHub - chunglabmit/pystripe: An image processing package for removing streaks from SPIM images.” https://github.com/chunglabmit/pystripe (accessed Dec. 12, 2023).

[14]	O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” CoRR, vol. abs/1505.04597, 2015, [Online]. Available: http://arxiv.org/abs/1505.04597

[15]	P. Zhang, W. Jiao, Y. Zhang, L. Li, and D. Zhang, “Improved U-Net for Industrial Image Denoising,” in 2023 CAA Symposium on Fault Detection, Supervision and Safety for Technical Processes (SAFEPROCESS), 2023, pp. 1–5. doi: 10.1109/SAFEPROCESS58597.2023.10295854.

[16]	D. Mehta, D. Padalia, K. Vora, and N. Mehendale, “MRI image denoising using U-Net and Image Processing Techniques,” in 2022 5th International Conference on Advances in Science and Technology (ICAST), 2022, pp. 306–313. doi: 10.1109/ICAST55766.2022.10039653.

[17]	J. Abascal, “GitHub - ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras.” https://github.com/jabascal/ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras (accessed Dec. 12, 2023).

[18]	S. van der Walt, “GitHub - _denoise.py.” https://github.com/scikit-image/scikit-image/blob/v0.22.0/skimage/restoration/_denoise.py#L254-L364 (accessed Dec. 12, 2023).

[19]	“OpenCV - Denoising Functions.” https://docs.opencv.org/3.4/d1/d79/group__photo__denoise.html (accessed Dec. 13, 2023).






