import streamlit as st
import pandas as pd
import re
import unicodedata
import emoji
import requests
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from pyvi import ViTokenizer
from collections import Counter

# ======================
# LOAD MODEL
# ======================
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("sentiment_model.pkl")

# ======================
# PREPROCESS
# ======================

SLANG_MAP = {
    "ko": "không",
    "k": "không",
    "kg": "không",
    "hok": "không",
    "hk": "không",
    "khum": "không",
    "kh": "không",
    "hem": "không",
    "khôg": "không",
    "khog": "không",


    "10đ": "10 điểm",
    "đỉm": "điểm",
    "dvien": "diễn viên",
    "dv": "diễn viên",
    "rv": "review",
    "ng": "người",
    "chớt": "chết",
    "cừ": "cười",
    "dỡ": "dở",
    "rcm": "recommend",
    "nv": "nhân vật",
    "nvat": "nhân vật",
    "nd": "nội dung",
    "thiệc": "thiệt",
    "nhãm": "nhảm",
    "đv": "đối với",
    "lun": "luôn",
    "mún": "muốn",

    
    "cũm": "cũng",
    "cx": "cũng",
    "vs": "với",
    "nma": "nhưng mà",
    "bth": "bình thường",
    "zui": "vui",
    "bngu": "buồn ngủ",
    "tr": "trời",
    "lquan": "liên quan",
    "tg": "thời gian",    

    "dc": "được",
    "đc": "được",
    "dk": "được",


    "oke": "ok",
    "okie": "ok",

    "vl": "vãi",
    "vcl": "vãi",
    "v": "vậy",


    "t": "tôi",       
    "mk": "mình",
    "mn": "mọi người",
    "mng": "mọi người",
    "củng": "cũng",
    "cũg": "cũng",

}

STOPWORDS = {
    "và","với","thì","là","mà","nhé","nhỉ","chứ","đấy","này","ấy","đó","thế","vậy",
    "ơ","ờ","ừ","à","đã","đang","rằng","trong","khi","nào","để","tại","từng",
    "ra","vào","ơi"
}

BLOCKED_WORDS = {"pass"}

# =====================================================
# 1. CLEAN + NORMALIZE 
# =====================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text).lower()

    # remove url
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # slang
    for k, v in SLANG_MAP.items():
        text = re.sub(rf"\b{k}\b", v, text)

    # newline + kéo dài
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # tách emoji
    text = "".join(
        f" {ch} " if ch in emoji.EMOJI_DATA else ch
        for ch in text
    )

    # bỏ dấu câu
    text = re.sub(r"[.,!?…;:()\"']", " ", text)

    # normalize space
    text = re.sub(r"\s+", " ", text).strip()

    words = [w for w in text.split() if w not in STOPWORDS]

    return " ".join(words)


# ==========================================


API_KEY = ""            # <--- THAY = API RIÊNG
VIDEO_ID = "h7RF-PBu-YM"          # <- ID video, vd: "dQw4w9WgXcQ"

import re

def extract_video_id(url):
    # case 1: youtube.com/watch?v=ID
    m = re.search(r"[?&]v=([^&]+)", url)
    if m:
        return m.group(1)
    
    # case 2: youtu.be/ID
    m = re.search(r"youtu\.be/([^?&]+)", url)
    if m:
        return m.group(1)
    
    return None

# ==========================================
# GET VIDEO INFO
# ==========================================
def get_video_info(video_id, api_key):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,statistics",
        "id": video_id,
        "key": api_key
    }

    data = requests.get(url, params=params).json()

    if not data["items"]:
        print("Không tìm thấy video.")
        return None

    info = data["items"][0]
    snippet = info["snippet"]
    stats = info["statistics"]

    return {
        "video_id": video_id,
        "video_title": snippet.get("title"),
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "channel_title": snippet.get("channelTitle"),
        "publishedAt": snippet.get("publishedAt"),
        "video_likeCount": stats.get("likeCount"),
        "video_commentCount": stats.get("commentCount")
    }


# ==========================================
# GET COMMENTS
# ==========================================
def get_youtube_comments(video_id, api_key, max_comments=4000, max_pages=1000):
    comments = []
    base_url = "https://www.googleapis.com/youtube/v3/commentThreads"

    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": api_key,
        "textFormat": "plainText",
        "maxResults": 100
    }

    page_count = 0

    while True:
        response = requests.get(base_url, params=params).json()

        if "items" not in response:
            print("Error:", response)
            break

        for item in response["items"]:
            c = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": c.get("authorDisplayName"),
                "text": c.get("textDisplay"),
                "likeCount": c.get("likeCount"),
                "publishedAt": c.get("publishedAt")
            })

            # DỪNG NGAY KHI ĐỦ 300 COMMENT
            if len(comments) >= max_comments:
                return comments

        page_count += 1
        print(f"Fetched page {page_count}, total comments: {len(comments)}")

        if "nextPageToken" in response and page_count < max_pages:
            params["pageToken"] = response["nextPageToken"]
        else:
            break

    return comments


# ==========================================
# RUN
# ==========================================

print("Fetching video info...")
video_info = get_video_info(VIDEO_ID, API_KEY)

print("Fetching comments...")
comments = get_youtube_comments(VIDEO_ID, API_KEY)

# Gộp thêm thông tin video vào từng comment
for c in comments:
    c.update(video_info)

# Tạo DataFrame
df = pd.DataFrame(comments)

# Lưu CSV
#df.to_csv("youtube_comments_full.csv", index=False, encoding="utf-8-sig")

# ======================
# PREDICT
# ======================
def predict_sentiment(comments):
    df = pd.DataFrame({"comment": comments})

    df["cleaned_text"] = df["comment"].apply(clean_text)
    df = df[~df["cleaned_text"].str.contains("|".join(BLOCKED_WORDS))] #filter

    #df["length"] = df["cleaned_text"].apply(lambda x: len(x.split()))
    #df["has_emoji"] = df["comment"].apply(has_emoji_fn)

    X_text = tfidf.transform(df["cleaned_text"])
    df["sentiment"] = model.predict(X_text)

    return df

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="YouTube Sentiment Analysis")

st.title("🎬 YouTube Movie Review Sentiment")

url = st.text_input("Nhập link YouTube")

if st.button("Phân tích"):
    video_id = extract_video_id(url)

    if video_id is None:
        st.error("❌ Link YouTube không hợp lệ")
    else:
        with st.spinner("Đang crawl & phân tích..."):
            # 1. Lấy info video
            video_info = get_video_info(video_id, API_KEY)

            if video_info is None:
                st.error("❌ Không lấy được thông tin video")
                st.stop()

            # 2. Hiển thị TITLE video
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
            st.image(
            thumbnail_url,
            caption="Thumbnail video",
            use_container_width=True
            )
            st.subheader("🎥 Thông tin video")
            st.markdown(f"**📌 Tiêu đề:** {video_info['video_title']}")
            st.markdown(f"**📺 Kênh:** {video_info['channel_title']}")
            st.markdown(f"**👍 Like:** {video_info['video_likeCount']}")
            st.markdown(f"**💬 Comment:** {video_info['video_commentCount']}")

            # 3. Lấy comment
            comments_raw = get_youtube_comments(video_id, API_KEY)

            comments_text = [c["text"] for c in comments_raw]

            # 4. Predict sentiment
            df_result = predict_sentiment(comments_text)

        st.success("✅ Phân tích xong!")

        # ======================
        # VISUALIZE
        # ======================
        def plot_wordcloud_by_label(df, label):
            text = " ".join(df[df["sentiment"] == label]["cleaned_text"])

            if text.strip() == "":
                st.info(f"Không đủ dữ liệu cho nhãn {label}")
                return

            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                collocations=False
            )

            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wc.generate(text))
            ax.axis("off")
            ax.set_title(f"WordCloud – {label}")

            st.pyplot(fig)

        def plot_sentiment_pie(df):
            counts = df["sentiment"].value_counts()

            fig, ax = plt.subplots()
            ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=90
            )
            ax.axis("equal")
            ax.set_title("Tỷ lệ cảm xúc bình luận")

            st.pyplot(fig)
        
        st.subheader("📊 Phân bố cảm xúc")
        st.bar_chart(df_result["sentiment"].value_counts())

        st.subheader("📊 Tỷ lệ cảm xúc (Pie chart)")
        plot_sentiment_pie(df_result)

        st.subheader("☁️ WordCloud theo nhãn cảm xúc")
        labels = ["positive", "mixed/neutral", "negative"]
        for lb in labels:
            plot_wordcloud_by_label(df_result, lb)

        st.subheader("💬 Kết quả chi tiết")
        st.dataframe(df_result[["comment", "sentiment"]])