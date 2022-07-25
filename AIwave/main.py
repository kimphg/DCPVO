import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
st.title("Nghiên cứu mạng nơ ron nhân tạo ANN dự đoán tham số sóng gần bờ khu vực Cửa Đại tỉnh Quảng Ngãi")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

    fig = px.line(        
            dataframe, #Data Frame
            x = [i for i in range(8738)], #Columns from the data frame
            y = "Gần bờ_Mike",
            title = "Line frame"
        )
    st.plotly_chart(fig)
    st.write(dataframe)
