import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from surprise import SVD, SVDpp, NMF, Dataset, Reader
from surprise.model_selection import cross_validate

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

# Load mô hình SVD đã lưu
with open('svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)

# # Hàm gợi ý khách sạn dựa trên người dùng
# def recommend_hotels(user_id, num_recommendations=5):
#     all_hotels = df_hotel_info['Hotel_ID'].unique()
#     rated_hotels = df_hotel_comments[df_hotel_comments['Reviewer ID'] == user_id]['Hotel ID'].unique()
#     unrated_hotels = [hotel for hotel in all_hotels if hotel not in rated_hotels]

#     predictions = [svd_model.predict(user_id, hotel).est for hotel in unrated_hotels]
#     recommendations = pd.DataFrame({
#         'Hotel_ID': unrated_hotels,
#         'Predicted_Score': predictions
#     })

#     top_recommendations = recommendations.sort_values(by='Predicted_Score', ascending=False).head(num_recommendations)
#     return df_hotel_info[df_hotel_info['Hotel_ID'].isin(top_recommendations['Hotel_ID'])][['Hotel_ID', 'Hotel_Name', 'Hotel_Rank', 'Hotel_Address', 'Total_Score']]

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

    # Chọn người dùng - Multiselect
    user_option = st.selectbox(
        "Chọn ID người dùng",
        df_hotel_comments['Reviewer ID'].unique(),
        index=None,
        placeholder="Ví dụ: 1_1_1",
    )

    # if user_option:
    #     user_id = user_option.split('\t')[0]  # Tách lấy User ID

    #     st.write('#### Các khách sạn gợi ý cho người dùng:')
    #     recommendations = recommend_hotels(user_id)
    #     st.dataframe(recommendations)

    #     if not recommendations.empty:
    #         # Chọn ID khách sạn với phần gợi ý
    #         hotel_id_selection = st.selectbox(
    #             "Chọn ID khách sạn để xem chi tiết",
    #             recommendations['Hotel_ID'],
    #             placeholder="Chọn một khách sạn"
    #         )
            
    #         # Lấy thông tin khách sạn
    #         selected_hotel = df_hotel_info[df_hotel_info['Hotel_ID'] == hotel_id_selection].iloc[0]
    #         st.write(f"Hotel Name: {selected_hotel['Hotel_Name']}")
    #         st.write(f"Hotel Rank: {selected_hotel['Hotel_Rank']}")
    #         st.write(f"Address: {selected_hotel['Hotel_Address']}")
    #         st.write(f"Total Score: {selected_hotel['Total_Score']}")
            
    #         description = selected_hotel['Hotel_Description']
            
    #         # Giới hạn mô tả xuống còn 500 từ
    #         limited_description = " ".join(description.split()[:500])
            
    #         # Hiển thị mô tả với giới hạn 500 từ
    #         if limited_description:
    #             st.write("Hotel Description:")
    #             with st.expander("Xem thêm"):
    #                 st.write(limited_description + "...")
    #         else:
    #             st.write("Không có mô tả cho khách sạn này.")
