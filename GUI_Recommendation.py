import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from st_aggrid import AgGrid, GridOptionsBuilder
import pickle

# load dữ liệu
df_hotel_comments = pd.read_csv('data/hotel_comments_cleaned.csv')
df_hotel_info     = pd.read_csv('data/hotel_info_cleaned.csv')

# Distinct hotel ID
df_hotel_id = df_hotel_info['Hotel_ID'] + '\t' + df_hotel_info['Hotel_Name']

# Tạo một ánh xạ từ tên khách sạn đến chỉ số
hotel_indices = pd.Series(df_hotel_info.index, index=df_hotel_info['Hotel_ID']).drop_duplicates()

# Đọc mô hình
# Open and read file to cosine_sim_new
with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

# Hàm lấy các khách sạn tương tự sử dụng độ tương đồng cosine
def get_similar_hotels_cosine(hotel_id, cosine_sim=cosine_sim, top_n=5):
    # Lấy chỉ số của khách sạn
    idx = hotel_indices[hotel_id]
    
    # Lấy điểm tương đồng cặp
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sắp xếp các khách sạn dựa trên điểm tương đồng
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Lấy điểm số của top n khách sạn tương tự nhất
    sim_scores = sim_scores[1:top_n+1]
    
    # Lấy chỉ số của các khách sạn
    hotel_indices_similar = [i[0] for i in sim_scores]
    
    # Trả về danh sách các khách sạn tương tự nhất
    # return df_hotel_info['Hotel_Name'].iloc[hotel_indices_similar]
    return df_hotel_info.iloc[hotel_indices_similar]

def print_hotel_details(hotel_info):
        st.write("Hotel ID: ", hotel_info['Hotel_ID'])
        st.write("Hotel Name: ", hotel_info['Hotel_Name'])
        st.write("Hotel Rank: ", hotel_info['Hotel_Rank'])
        st.write("Hotel Address: ", hotel_info['Hotel_Address'])
        st.write("Total Score: ", hotel_info['Total_Score'])
        st.write("Location: ", hotel_info['Location'])
        st.write("Cleanliness:  ", hotel_info['Cleanliness'])
        st.write("Service:  ", hotel_info['Service'])
        st.write("Facilities:  ", hotel_info['Facilities'])
        st.write("Value for money:  ", hotel_info['Value_for_money'])
        st.write("Comments count:  ", hotel_info['comments_count'])
        # st.write("Hotel Description:  ", hotel_info['Hotel_Description'])
        # with st.expander("Hotel Description"):
        #     st.write(hotel_info['Hotel_Description'])

        st.write("Hotel Description: ")
        # Hiển thị văn bản rút gọn với st.expander
        with st.expander("Xem thêm"):
            st.write(hotel_info['Hotel_Description'])

#################################################################
# Thêm tiêu đề vào sidebar
st.sidebar.header("ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE")
st.sidebar.markdown("#### Recommendation System")
st.sidebar.write('HV1: _Thang Tuấn Văn_')
st.sidebar.write('HV2: _Nguyễn Xuân Trường_')
st.sidebar.divider()

menu = ["Content Based", "Collaborative Filtering"]
choice = st.sidebar.selectbox('Menu', menu)

st.sidebar.divider()

st.sidebar.write("""#### Giới thiệu về project
- Xây dựng hệ thống đề xuất để hỗ trợ người dùng nhanh chóng chọn được nơi lưu trú phù hợp trên Agoda → Hệ thống sẽ gồm hai mô hình gợi ý chính:
    - Content-based filtering
    - Collaborative filtering""")

if choice == 'Content Based':  
    st.subheader("Content Based")

    # Chọn khách sạn - Multiselect
    option = st.selectbox(
        "Nhập ID khách sạn",
        df_hotel_id,
        index=None,
        placeholder="Ví dụ: 1_1",
    )

    if option != None:
        hotel_id = option.split('\t')[0]

        # Hiển thị thông tin của khách sạn vừa chọn
        st.write('#### Bạn vừa chọn:')
        
        selected_hotel = df_hotel_info[df_hotel_info['Hotel_ID'] == hotel_id]

        hotel_info = selected_hotel.iloc[0]

        print_hotel_details(hotel_info)

        st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
        # Hiển thị recommend - Sử dụng Hotel ID để lấy tên khách sạn
        if hotel_id in hotel_indices:
            similar_hotels_cosine = get_similar_hotels_cosine(hotel_id)
            
            # st.write(similar_hotels_cosine)
            df_similar_hotels = similar_hotels_cosine[['Hotel_ID','Hotel_Name']]
          
            # Configure the table with AgGrid
            gb = GridOptionsBuilder.from_dataframe(df_similar_hotels)
            gb.configure_selection('single')  # 'single' allows selecting only one row at a time
            grid_options = gb.build()

            # Display the table
            grid_response = AgGrid(
                df_similar_hotels,
                gridOptions=grid_options,
                height=200,
                width='100%',
            )

            # Get the selected row
            selected_row = grid_response['selected_rows']
            if selected_row is not None:
                # Display selected row
                selected_hotel = df_hotel_info[df_hotel_info['Hotel_ID'] == selected_row.iloc[0]['Hotel_ID']]
                hotel_info = selected_hotel.iloc[0]
                st.write('##### Thông tin khách sạn:')
                print_hotel_details(hotel_info)

        else:
            st.write("Hotel ID không tồn tại.")

elif choice == "Collaborative Filtering":
    st.subheader("Collaborative Filtering")