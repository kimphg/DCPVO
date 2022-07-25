import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
df = pd.read_csv(
  "data_ann.csv"
)

st.title("Nghiên cứu mạng nơ ron nhân tạo ANN dự đoán tham số sóng gần bờ khu vực Cửa Đại tỉnh Quảng Ngãi")

st.write("Du lieu:") 

fig = px.line(        
        df, #Data Frame
        x = [i for i in range(8738)], #Columns from the data frame
        y = "Gần bờ_Mike",
        title = "Line frame"
    )
st.plotly_chart(fig)
st.write(df)
