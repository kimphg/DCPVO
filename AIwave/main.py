from audioop import avg
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tsod
import streamlit as st
import plotly.express as px
from luminaire.exploration.data_exploration import DataExploration
from luminaire.model.lad_filtering import LADFilteringModel

st.title("Đề tài: Nghiên cứu mạng nơ ron nhân tạo ANN dự đoán tham số sóng gần bờ khu vực Cửa Đại tỉnh Quảng Ngãi")
st.header("Phần mềm phân tích dữ liệu bằng mạng nơ ron nhân tạo ANN")
uploaded_file = st.file_uploader(label="Choose a file", 
                                accept_multiple_files=False, 
                                help="Vui lòng tải file dữ liệu .csv theo đúng định dạng để chương trình có thể xử lý, cột thời gian phải  ở đầu tiên, và hàng đầu tiên trong file là tên các cột",
                                type=['csv'])


if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
    st.header("Dữ liệu đã tải lên")
    st.subheader("Số bản ghi: "+  str(dataframe.shape[0]))
    st.write(dataframe)


    # dataframe = pd.read_csv("data_test.csv", parse_dates=True, index_col=0)
    # print(df.shape)
    st.header("Chọn trường dữ liệu")
    data_columns = ['Chiều cao', 'Chu kỳ', 'Hướng']
    is_ANN_model_processed = [False, False, False]

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

        if show_options == 'Vẽ biểu đồ dữ liệu':
            st.subheader('Biểu đồ dữ liệu')
            # values = st.slider(
            #                     'Chọn ngưỡng để lọc',
            #                     0.0, float(int(max(dataframe[data_columns[id]])+3.0)), (0.0, 3.5), step=0.1)
            values = st.slider(
                                'Chọn ngưỡng để lọc',
                                0.0, float(int(max(dataframe[data_columns[id]])+3.0)), (float(0.0), float(dataframe[data_columns[id]].mean()*1.5)), step=0.1)

            rd = tsod.RangeDetector(min_value=values[0], max_value=values[1])
            res = rd.detect(dataframe[data_columns[id]])
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=dataframe[data_columns[id]], mode='lines', line=dict(color = 'Steelblue'), name='Bình thường'))
            fig.add_trace(go.Scatter(y=dataframe[data_columns[id]].where(res==True), mode='lines', line=dict(color = 'red'), name='Vượt ngưỡng'))
            st.plotly_chart(fig)

        elif show_options == 'Phát hiện dao động mạnh':
            st.subheader('Biểu đồ phân tích dao động')
            magd = tsod.GradientDetector()
            magd.fit(dataframe[data_columns[id]][0:int(dataframe[data_columns[id]].shape[0]/20)])
            res = magd.detect(dataframe[data_columns[id]])
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=dataframe[data_columns[id]], mode='lines', line=dict(color = 'Steelblue'), name='Bình thường'))
            fig.add_trace(go.Scatter(y=dataframe[data_columns[id]].where(res==True), mode='lines', line=dict(color = 'red'), name='Điểm dao động'))
            st.plotly_chart(fig)
        
        elif show_options == 'Dự đoán chuỗi dữ liệu bằng học máy' and not is_ANN_model_processed[id]:
            # dataframe = pd.read_csv("data_test.csv", parse_dates=True, index_col=0)
            st.subheader('Biểu đồ dự đoán chuỗi dữ liệu')
            raw_data = dataframe[[data_columns[id]]]
            raw_data.rename(columns = {data_columns[id]:'raw'}, inplace = True)
            de_obj = DataExploration(freq='D', data_shift_truncate=False, is_log_transformed=True, fill_rate=0.9)
            raw_data, pre_prc = de_obj.profile(raw_data)
            hyper_params = {"is_log_transformed": 1}
            lad_filter_obj = LADFilteringModel(hyper_params=hyper_params, freq='D')
            print('Dang huan luyen mo hinh')
            success, model_date, model = lad_filter_obj.train(data=raw_data, **pre_prc)

            data_org = dataframe[[data_columns[id]]]
            data_org_1 = dataframe[[data_columns[id]]]
            data_pred = dataframe[[data_columns[id]]]
            data_anormal = data_org[data_columns[id]]

            print('Dang du doan du lieu')
            scores, model_update = model.score(data_org[data_columns[id]][0], data_org[data_columns[id]].index[0])
            for i in range(data_org.shape[0]):
                scores, model_update = model_update.score(data_org[data_columns[id]][i], data_org[data_columns[id]].index[i])
                data_pred[data_columns[id]][i] = scores['Prediction']
                if scores['IsAnomaly'] == True:
                    data_anormal[data_org[data_columns[id]].index[i]] = True
                else:
                    data_anormal[data_org[data_columns[id]].index[i]] = False

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=data_pred[data_columns[id]], mode='lines', line=dict(color = 'Steelblue'), name='Dự đoán'))
            fig.add_trace(go.Scatter(y=data_org_1[data_columns[id]], mode='lines', line=dict(color = 'Green'), name='Thực tế'))
            fig.add_trace(go.Scatter(y=data_org_1[data_columns[id]].where(data_anormal==True), mode='markers', line=dict(color = 'red'), name='Điểm dao độn'))
            st.plotly_chart(fig)
            is_ANN_model_processed = True

        else:
            pass

st.write("[Trở về trang chủ](http://aiwave.vn)")