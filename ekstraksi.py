import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops

dataset_path = "dataset"

nama_data = []
warna_data = []
tekstur_data = []
bentuk_data = []

for label in os.listdir(dataset_path):

    folder = os.path.join(dataset_path, label)

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path)
        img = cv2.resize(img,(200,200))

        # ===== WARNA =====
        mean_color = img.mean(axis=(0,1))
        B,G,R = mean_color

        std_color = img.std(axis=(0,1))
        B_std,G_std,R_std = std_color

        R_skew = np.mean((img[:,:,2] - R)**3)
        G_skew = np.mean((img[:,:,1] - G)**3)
        B_skew = np.mean((img[:,:,0] - B)**3)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        H_mean = hsv[:,:,0].mean()
        S_mean = hsv[:,:,1].mean()
        V_mean = hsv[:,:,2].mean()

        H_std = hsv[:,:,0].std()
        S_std = hsv[:,:,1].std()
        V_std = hsv[:,:,2].std()

        RG_ratio = R/(G+1e-5)
        RB_ratio = R/(B+1e-5)
        GB_ratio = G/(B+1e-5)


        # ===== TEKSTUR =====
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        glcm = graycomatrix(gray,[1],[0],256, symmetric=True, normed=True)

        contrast = graycoprops(glcm,'contrast')[0,0]
        homogeneity = graycoprops(glcm,'homogeneity')[0,0]
        energy = graycoprops(glcm,'energy')[0,0]
        correlation = graycoprops(glcm,'correlation')[0,0]

        dissimilarity = graycoprops(glcm,'dissimilarity')[0,0]
        ASM = graycoprops(glcm,'ASM')[0,0]

        gray_mean = gray.mean()
        gray_std = gray.std()

        glcm_prob = glcm / glcm.sum()
        entropy = -np.sum(glcm_prob * np.log2(glcm_prob + 1e-10))

        glcm_var = np.var(glcm)


        # ===== BENTUK =====
        _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours)>0:
            c=max(contours,key=cv2.contourArea)

            area=cv2.contourArea(c)
            perimeter=cv2.arcLength(c,True)

            x,y,w,h = cv2.boundingRect(c)
            aspect_ratio = float(w)/h

            rect_area = w*h
            extent = float(area)/rect_area

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)

            solidity = area / (hull_area + 1e-5)

            circularity = (4*np.pi*area)/(perimeter**2 + 1e-5)

            eq_diameter = np.sqrt(4*area/np.pi)

            rectangularity = area/(rect_area+1e-5)

        else:
            area=0
            perimeter=0
            aspect_ratio=0
            extent=0
            solidity=0
            circularity=0
            eq_diameter=0
            rectangularity=0


        # ===== SIMPAN =====
        nama_data.append([file,label])

        warna_data.append([
            file,
            R,G,B,
            R_std,G_std,B_std,
            H_mean,S_mean,V_mean,
            H_std,S_std,V_std,
            R_skew,G_skew,B_skew,
            RG_ratio,RB_ratio,GB_ratio
        ])

        tekstur_data.append([
            file,
            contrast,homogeneity,energy,correlation,
            dissimilarity,ASM,
            gray_mean,gray_std,
            entropy,glcm_var
        ])

        bentuk_data.append([
            file,
            area,perimeter,aspect_ratio,extent,
            solidity,circularity,eq_diameter,rectangularity
        ])


# ===== DATAFRAME =====
df_nama = pd.DataFrame(nama_data,columns=["file","class"])

df_warna = pd.DataFrame(warna_data,columns=[
    "file",
    "R_mean","G_mean","B_mean",
    "R_std","G_std","B_std",
    "H_mean","S_mean","V_mean",
    "H_std","S_std","V_std",
    "R_skew","G_skew","B_skew",
    "RG_ratio","RB_ratio","GB_ratio"
])

df_tekstur = pd.DataFrame(tekstur_data,columns=[
    "file",
    "contrast","homogeneity","energy","correlation",
    "dissimilarity","ASM",
    "gray_mean","gray_std",
    "entropy","glcm_var"
])

df_bentuk = pd.DataFrame(bentuk_data,columns=[
    "file",
    "area","perimeter","aspect_ratio","extent",
    "solidity","circularity","eq_diameter","rectangularity"
])


with pd.ExcelWriter("fitur_dataset.xlsx") as writer:
    df_nama.to_excel(writer,sheet_name="nama",index=False)
    df_warna.to_excel(writer,sheet_name="warna",index=False)
    df_tekstur.to_excel(writer,sheet_name="tekstur",index=False)
    df_bentuk.to_excel(writer,sheet_name="bentuk",index=False)

print("Ekstraksi selesai")