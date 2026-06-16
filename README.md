# **CFM-Rx: Score-Based Conditional Flow Models for MIMO Receiver Design with Superimposed Pilots**

This repository contains the official implementation of the paper:

**Score-Based Conditional Flow Models for MIMO Receiver Design with Superimposed Pilots** *Published in IEEE Open Journal of the Communications Society*

**Authors:** Ruhao Zhang, Yupeng Li, Yitong Liu, Shijian Gao, Jing Jin, Hongwen Yang, and Jiangzhou Wang.

## **📌 Abstract**

This paper proposes a **Conditional Flow Matching Receiver (CFM-Rx)**, an unsupervised generative framework for joint channel estimation and data detection in MIMO systems with **Superimposed Pilots (SIP)**.

Key features include:

* **Unsupervised Learning:** Learns directly from received signals without labeled bit streams.  
* **Deterministic Inference:** Uses ODE-based sampling instead of stochastic SDE, reducing latency.  
* **Robustness:** Specifically designed to handle pilot contamination in SIP configurations.  
* **Performance:** Outperforms conventional LMMSE and diffusion-based receivers (DM-JED).

## **🛠️ Dependencies**

This code is built on **Python 3.8** and **PyTorch 2.3.1**.

To install the required libraries, run:

pip install \-r requirements.txt

## **🚀 Usage**

### **1\. Data Preparation**

The simulations use the **3GPP Clustered Delay Line (CDL)** channel model.

Please download the dataset from the following link and place it in the appropriate data folder:  
[Dataset Download Link](https://drive.google.com/drive/folders/1MXIupDTuUW69pTW3iXi5By59Ikk61uiz?usp=drive_link)

### **2\. Training**

To train the CFM-Rx model under the SIP configuration, run the following command.  
Note: Training parameters (e.g., epochs, batch size, SNR range) need to be configured directly inside run\_diffCNN.py.  
python run\_diffCNN.py

### **3\. Inference & Evaluation**

To evaluate the model (BER/NMSE) using the predictor-corrector sampler, run the following command.  
Note: Inference parameters (e.g., checkpoint path, sampling steps) need to be configured directly inside test\_flow.py.  
python test\_flow.py

## **📊 Parameters & Configuration**

The default implementation follows the experimental setup in the paper:

| Parameter | Value |
| :---- | :---- |
| **MIMO Size** | $4 \\times 4$ |
| **Channel Model** | 3GPP CDL-C (Delay spread: 300ns, Speed: 3km/h) |
| **Pilot Scheme** | Superimposed Pilots (SIP) ($w=0.9, v=0.1$) |
| **Modulation** | QPSK (Default), 8PSK, 16PSK |
| **Sampling Steps (**$T$**)** | 30 |
| **Corrector Steps (**$K$**)** | 5 |

