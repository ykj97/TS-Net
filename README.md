# TS-Net

Tumor segmentation-Net (TS-Net) is a breast tumor segmentation model for an orthotopic breast cancer model using MRI.


TS-Net involves replacing the encoder of U-Net with a pre-trained ResNet34 to improve performance.
The model was trained using a sample size of n=19 from the untreated group and then subsequently assessed on both the untreated group (n=5) and Doxorubicin (DOX)-treated group (n=6). Accoring to this strategy, the dataset with untreated and DOX-treated could be utilized for your purpose. 

## Pipeline of TS-Net
![jun1](https://github.com/ykj97/TS-Net/assets/131689170/7872eec2-f55c-499f-a783-abca0b1bd65d)



