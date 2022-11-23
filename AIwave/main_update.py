from audioop import avg
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import tsod
import streamlit as st
import plotly.express as px
from luminaire.exploration.data_exploration import DataExploration
from luminaire.model.lad_filtering import LADFilteringModel

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
# from statsmodels.tsa.vector_ar.var_model import VAR


## pip install statsmodels==0.12.2
## pip install kats
from kats.consts import TimeSeriesData
from kats.models.var import VARModel, VARParams
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

st.title("Nghiên cứu mạng nơ ron nhân tạo ANN dự đoán tham số sóng gần bờ khu vực Cửa Đại tỉnh Quảng Ngãi")

uploaded_file = st.file_uploader(label="Choose a file", 
                                accept_multiple_files=False, 
                                help="Vui lòng tải file dữ liệu .csv theo đúng định dạng để chương trình có thể xử lý",
                                type=['csv'])


if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
    dataframe = dataframe.fillna(0)
    st.header("Dữ liệu đã tải lên")
    st.subheader("Số bản ghi: "+  str(dataframe.shape[0]))
    st.write(dataframe)


    # dataframe = pd.read_csv("data_test.csv", 	parse_dates=True, index_col=0)
    # print(df.shape)
    st.header("Chọn trường dữ liệu ")
    data_columns = ['Chiều cao', 'Chu kỳ', 'Hướng', 'Chiều cao gần bờ', 'Chu kỳ gần bờ', 'Hướng gần bờ']

    field = st.radio(
                "Chọn trường dữ liệu để phân tích",
                ('Chiều cao', 'Chu kỳ', 'Hướng'),key='radio_master')
    id = data_columns.index(field)

    # id = 0
    # for id in range(len(data_columns)):
    if data_columns[id] in dataframe.columns:
        st.header("Phân tích dữ liệu " + data_columns[id])
        show_options = st.radio(
            "Chọn để phân tích",
            ('Vẽ biểu đồ dữ liệu', 'Phát hiện dao động mạnh', 'Dự đoán chuỗi dữ liệu bằng học máy'),key=data_columns[id])
        id_near = id + 3
        if show_options == 'Vẽ biểu đồ dữ liệu':
            st.subheader('Biểu đồ dữ liệu')
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=dataframe[data_columns[id]], mode='lines', line=dict(color = 'Steelblue'), name='Xa bờ'))
            fig.add_trace(go.Scatter(y=dataframe[data_columns[id_near]], mode='lines', line=dict(color = 'green'), name='Gần bờ'))
            st.plotly_chart(fig)

        elif show_options == 'Phát hiện dao động mạnh':
            st.subheader('Biểu đồ phân tích dao động')
            st.caption('Dữ liệu ' + data_columns[id] + ' xa bờ: ')
            magd = tsod.GradientDetector()
            magd.fit(dataframe[data_columns[id]][0:int(dataframe[data_columns[id]].shape[0]/20)])
            res = magd.detect(dataframe[data_columns[id]])
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=dataframe[data_columns[id]], mode='lines', line=dict(color = 'Steelblue'), name='Bình thường'))
            fig.add_trace(go.Scatter(y=dataframe[data_columns[id]].where(res==True), mode='lines', line=dict(color = 'red'), name='Điểm dao động'))
            st.plotly_chart(fig)
            st.caption('Dữ liệu ' + data_columns[id] + ' gần bờ: ')
            magd = tsod.GradientDetector()
            magd.fit(dataframe[data_columns[id]][0:int(dataframe[data_columns[id_near]].shape[0]/20)])
            res = magd.detect(dataframe[data_columns[id_near]])
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=dataframe[data_columns[id_near]], mode='lines', line=dict(color = 'Steelblue'), name='Bình thường'))
            fig.add_trace(go.Scatter(y=dataframe[data_columns[id_near]].where(res==True), mode='lines', line=dict(color = 'red'), name='Điểm dao động'))
            st.plotly_chart(fig)
        
        elif show_options == 'Dự đoán chuỗi dữ liệu bằng học máy':
            # dataframe = pd.read_csv("data_test.csv", parse_dates=True, index_col=0)
            st.subheader('Biểu đồ dự đoán chuỗi dữ liệu')
            
            multi_df = dataframe[[data_columns[id], data_columns[id_near]]].copy()
            # multi_df = multi_df[:2000].copy()
            multi_df.reset_index(inplace=True)
            multi_df = multi_df.rename(columns={"Thời gian": "time"})

            split_point = int(0.9*len(multi_df))
            multi_df_train, multi_df_val = multi_df[:split_point], multi_df[split_point:]

            multi_ts = TimeSeriesData(multi_df_train)
            params = VARParams()
            maxlags = int(len(multi_df_train)*0.2)
            # maxlags = 300
            params.maxlags = maxlags
            params.trend = 'ctt'
            # params.ic = 'bic'

            m = VARModel(multi_ts, params)
            m.fit(verbose=True)
            fcst = m.predict(steps=int(len(multi_df_val)))

            rmse = sqrt(mean_squared_error(multi_df_val['Chiều cao'].values, fcst['Chiều cao']['fcst'].value))
            rmse_gb = sqrt(mean_squared_error(multi_df_val['Chiều cao gần bờ'].values, fcst['Chiều cao gần bờ']['fcst'].value))


            st.caption('Dự đoán dữ liệu ' + data_columns[id_near] + ' xa bờ')
            st.caption('RMSE: ' + str(rmse))
            fig = go.Figure()
            to_plot = np.concatenate((multi_df_train[data_columns[id]].to_numpy(), fcst[data_columns[id]]['fcst'].value), axis=0)
            samples_predict = len(multi_df_val)
            fig.add_trace(go.Scatter(y=to_plot[-2*samples_predict:], mode='lines', line=dict(color = 'Green'), name='Dự đoán'))
            fig.add_trace(go.Scatter(y=multi_df[data_columns[id]].values[-2*samples_predict:], mode='lines', line=dict(color = 'Steelblue'), name='Dữ liệu thực tế'))
            # fig.add_trace(go.Scatter(y=multi_df[data_columns[id]].values, mode='lines', line=dict(color = 'Red'), name='Dữ liệu xa bờ'))
            st.plotly_chart(fig)


            st.caption('Dự đoán dữ liệu ' + data_columns[id_near])
            st.caption('RMSE: ' + str(rmse_gb))
            fig = go.Figure()
            to_plot = np.concatenate((multi_df_train[data_columns[id_near]].to_numpy(), fcst[data_columns[id_near]]['fcst'].value), axis=0)
            samples_predict = len(multi_df_val)
            fig.add_trace(go.Scatter(y=to_plot[-2*samples_predict:], mode='lines', line=dict(color = 'Green'), name='Dự đoán'))
            fig.add_trace(go.Scatter(y=multi_df[data_columns[id_near]].values[-2*samples_predict:], mode='lines', line=dict(color = 'Steelblue'), name='Dữ liệu thực tế'))
            # fig.add_trace(go.Scatter(y=multi_df[data_columns[id]].values, mode='lines', line=dict(color = 'Red'), name='Dữ liệu xa bờ'))
            st.plotly_chart(fig)

            # print(fcst[data_columns[id_near]]['fcst'].value)
            # print(type(multi_df_val[data_columns[id_near]].values))

        else:
            pass

st.write("[Trở về trang chủ](http://aiwave.vn)")
