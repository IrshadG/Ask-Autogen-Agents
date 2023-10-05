#python RunPOP/GTM_Deployer.py
############################ IMPORT DEPENDENCIES ##############################

import streamlit as st
import gradio as gr
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from models.GTM import GTM
from utils.data_multitrends import ZeroShotDataset
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

############################ CALCULATE ERROR ##############################
def cal_error_metrics(gt, forecasts):
    # Absolute errors
    mae = mean_absolute_error(gt, forecasts)
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)

    return round(mae, 3), round(wape, 3)


def get_key_from_dict(dict , vals):
    # list out keys and values separately
    key_list = list(dict.keys())
    val_list = list(dict.values())
     
    # Get key with val 
    keys = []
    for val in vals:
        keys.append(key_list[val_list.index(val)])
    return keys

########################### DEFINE ARGUMENTS ##########################
class Args():
    def __init__(self):
        self.data_folder = 'RunPOP/dataset/'
        self.ckpt_path = 'RunPOP/log/GTM/GTM_epoch35.ckpt'
        self.gpu_num = 0
        self.seed = 21
    
        self.model_type = 'GTM'
        self.use_trends = 1
        self.use_img = 1
        self.use_text = 1
        self.trend_len = 52
        self.num_trends = 3
        self.embedding_dim =32 
        self.hidden_dim = 64
        self.output_dim = 12
        self.use_encoder_mask =1
        self.autoregressive = 0
        self.num_attn_heads = 4
        self.num_hidden_layers = 1
    
        self.wandb_run = 'GTM_Deploy'   

############################# LOAD DATA ############################

#Create args
args = Args()

# Set up CUDA
device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

# Seeds for reproducibility
pl.seed_everything(args.seed)

# Load sales data    
test_df = pd.read_csv(Path(args.data_folder + 'test.csv'), parse_dates=['release_date'])
item_codes = test_df['external_code'].values
img_paths = test_df['image_path'][5:]

 # Load category and color encodings
cat_dict = torch.load(Path(args.data_folder + 'category_labels.pt'))
col_dict = torch.load(Path(args.data_folder + 'color_labels.pt'))
fab_dict = torch.load(Path(args.data_folder + 'fabric_labels.pt'))

# Load Google trends
gtrends = pd.read_csv(Path(args.data_folder + 'gtrends.csv'), index_col=[0], parse_dates=True)

def load_dataloader():
    test_loader = ZeroShotDataset(test_df, Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict, \
            fab_dict, args.trend_len).get_loader(batch_size=1, train=False)
    return test_loader


# model_savename = f'{args.wandb_run}_{args.output_dim}'

############################# CREATE MODEL ############################
def load_Model():
    model = GTM(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_attn_heads,
            num_layers=args.num_hidden_layers,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            autoregressive=args.autoregressive,
            gpu_num=args.gpu_num
        )

    #Load from loaded state
    model.load_state_dict(torch.load(args.ckpt_path)['state_dict'], strict=False)
    return model


############################# FORECAST SALES ############################

def predict_sales(img_name):
    n_imgs = []

    #Fix the path names
    for i in img_paths:
        n_imgs.append(i[5:])

    index = n_imgs.index(img_name)
    #st.write("Index is: " + str(index))
    test_loader = load_dataloader()

    category = get_key_from_dict(cat_dict, [test_loader.dataset[index][1].numpy()])[0]
    color = get_key_from_dict(col_dict, [test_loader.dataset[index][2].numpy()])[0]
    fabric = get_key_from_dict(fab_dict, [test_loader.dataset[index][3].numpy()])[0]

    st.divider()
    st.image(uploaded_file, width=168, caption='Uploaded image')
    st.write("Category - " + str(category))
    st.write("Color - " + str(color))
    st.write("Fabric - " + str(fabric))

    #Prediction
    test_data = test_loader.dataset[index]
    model = load_Model()
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_data = [tensor.to(device) for tensor in test_data]
        item_sales, category, color, textures, temporal_features, gtrends, images =  test_data
        y_pred, att = model(category[None], color[None],textures[None], temporal_features[None], gtrends[None], images[None])

        forecasts = y_pred.detach().cpu().numpy().flatten()[:args.output_dim]
        gt = item_sales.detach().cpu().numpy().flatten()[:args.output_dim]
        attns = att.detach().cpu().numpy()

    rescale_vals = np.load(args.data_folder + 'normalization_scale.npy')
    rescaled_forecasts = forecasts * rescale_vals
    rescaled_gt = gt * rescale_vals

    mae, wape = cal_error_metrics(gt, forecasts)
    rescaled_mae, rescaled_wape = cal_error_metrics(rescaled_gt, rescaled_forecasts)
    st.divider()
    st.header('Prediction')
    # plot lines
    st.line_chart(pd.DataFrame({"Actual Sales":rescaled_gt,
                                "Predicted Sales":rescaled_forecasts}))

    st.divider()
    st.header('Metrics')
    st.write("Mean Absolute Error: ",mae)
    st.write("Weighted Absolute Percentage Error: ", wape)
    st.write("Rescaled-Mean Absolute Error: ", rescaled_mae)
    st.write("Rescaled-Weighted Absolute Percentage Error", rescaled_wape)



#Get Image for prediction
st.title('AI Sales Prediction Model')
st.header('Image Details')
file_name=""
uploaded_file = st.file_uploader("Choose an image file", accept_multiple_files=False)
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    file_name = uploaded_file.name
    predict_sales(file_name)

