import csv
import time
from selenium import webdriver
from selenium_stealth import stealth
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.webdriver import WebDriver
import pandas as pd
import json
import re

# Khởi tạo WebDriver với cấu hình tối ưu
def init_driver():
    chrome_driver_path = "/Users/thuyduc/Downloads/chromedriver-mac-arm64/chromedriver"  # Đường dẫn đến ChromeDriver
    service = Service(chrome_driver_path)
    options = Options()
    
    # Cấu hình trình duyệt
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")
    options.add_argument("--headless=new")  # Chế độ ẩn trình duyệt
    
    driver = webdriver.Chrome(service=service, options=options)
    # Tăng thời gian chờ tải trang
    # driver.set_page_load_timeout(360)  # 300 giây (5 phút)

    
    # Tăng thời gian chờ tìm phần tử
    # driver.implicitly_wait(1)  # 30 giây
    # Kích hoạt Selenium Stealth để tránh bị phát hiện
    stealth(driver,
            languages=["en-US", "en"],
            vendor="Apple Inc.",
            platform="MacARM",
            webgl_vendor="Apple Inc.",
            renderer="Apple GPU",
            fix_hairline=True)
    
    return driver

def get_film_review_url(driver: WebDriver, base_url:str) -> list[dict]|None:
    driver.get(base_url)
    specific_urls_list = []
    try:
        # Chờ cho phần tử chứa liên kết đánh giá xuất hiện
        container = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((
                By.ID,
                "review"
            ))
        )

        for _ in range(60):  # Lặp để nhấn nút "Xem tiếp nhé !" nhiều lần
            try:
                xemthem_button = container.find_element(By.XPATH, "//button[contains(text(), 'Xem tiếp nhé !')]")
                # xemthem_button.click()
                driver.execute_script("arguments[0].click();", xemthem_button)
                time.sleep(2)  # Chờ một chút để nội dung tải thêm
            except TimeoutException:
                print("Không tìm thấy nút 'Xem tiếp nhé !' hoặc không thể nhấn.")
                break

         
        # Lấy tất cả các liên kết đánh giá sau khi đã nhấn nút nhiều lần
        review_grid = container.find_element(By.CSS_SELECTOR, "div.grid.grid-cols-1.gap-x-5.gap-y-6.md\\:grid-cols-2.lg\\:grid-cols-3")
        
        flag, count = True, 1
        while flag:
            try:
                review = review_grid.find_element(By.XPATH, f"//*[@id=\"review\"]/div/div[2]/div[{count}]/div")
                
                dict_review = {}
                try:
                    overall_rating = review.find_element(By.CSS_SELECTOR, "div.flex.items-center.space-x-1.text-sm.text-white").text
                except Exception as e:
                    print(f"Phát hiện lỗi ở liên kết đánh giá thứ {count}")
                    print("Bỏ qua liên kết này...")
                    count += 1
                    continue

                review_section = review.find_element(By.CSS_SELECTOR, "a[href*='/review']")
                href = review_section.get_attribute("href")
                title = review.find_element(By.CSS_SELECTOR, "div.truncate.text-sm.font-bold.leading-tight.text-white.hover\\:text-pink-100").text
                num_comments = review_section.text

                if re.search(r'K$', num_comments, flags=re.IGNORECASE):
                    num_comments = re.sub(r'K$', '', num_comments, flags=re.IGNORECASE)
                    num_comments = float(num_comments) * 1000
                else:
                    num_comments = re.sub(r'[^0-9]', '', num_comments)
                    num_comments = float(num_comments)

                overall_rating = float(re.sub(r'[^0-9.]', '', overall_rating))

                dict_review['title'] = title
                dict_review['overall_rating'] = overall_rating
                dict_review['num_comments'] = num_comments
                dict_review['url'] = href
                
                specific_urls_list.append(dict_review)
                count += 1
            except Exception as e:
                # print(f"Lỗi khi lấy liên kết đánh giá thứ {count}: {e}")
                flag = False

        return specific_urls_list
    except TimeoutException:
        print("Không tìm thấy liên kết đánh giá.")
        return None
    
def get_specific_film_review(driver: WebDriver, review_urls_dict: list[dict]) -> list[dict]:
    total_data = []
    
    for review_url_dict in review_urls_dict:
        review_url = review_url_dict['url']
        review_title = review_url_dict['title']
        review_overall_rating = review_url_dict['overall_rating']
        review_num_comments = review_url_dict['num_comments']

        driver.get(review_url) # Truy cập vào trang đánh giá cụ thể
        print(f"Đang lấy đánh giá từ: {review_url}")
        dict_review = {}
        
        dict_review['author'] = []
        dict_review['comment'] = []
        dict_review['ratings'] = []
        dict_review['title'] = []
        dict_review['url'] = []
        
        try:
            # Chờ cho phần tử chứa nội dung đánh giá xuất hiện
            container = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    "div.mx-auto.w-full.max-w-6xl.px-5.md\\:px-8.lg\\:px-8.py-4.md\\:py-8"
                ))
            )
            review_content = container.find_element(By.CSS_SELECTOR, "div[class*='review-content']")
            
            # Nhấn nút "Xem tiếp nhé!" để tải thêm nội dung đánh giá
            for _ in range(10):  # Lặp để nhấn nút nhiều lần
                try:
                    # Lấy nút xem thêm
                    xemthem_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Xem tiếp nhé!')]"))
                    )
                    # print(xemthem_button.text)
                    driver.execute_script("arguments[0].click();", xemthem_button)
                    # xemthem_button.click()
                    time.sleep(2)  # Chờ một chút để nội dung tải thêm
                except TimeoutException:
                    print("Không tìm thấy nút 'Xem tiếp nhé!' hoặc không thể nhấn.")
                    break
            
            review_section = review_content.find_element(By.CSS_SELECTOR, "div.grid.grid-cols-1.divide-y.divide-gray-200")
                
            # Lấy đánh giá sao
            ratings = review_section.find_elements(By.CSS_SELECTOR, "div.flex.items-center.text-md.-ml-1.mb-0\\.5.font-semibold.text-gray-900")
            for rating in ratings:
                texts = rating.find_elements(By.CSS_SELECTOR, "span")
                text = [t.text for t in texts]
                text = "".join(text)
                dict_review['ratings'].append(text)
            
            # Lấy tác giả đánh giá
            authors = review_section.find_elements(By.CSS_SELECTOR, "div.text-md.text-gray-800")
            for author in authors:
                dict_review['author'].append(author.text)

            # Lấy nội dung đánh giá
            comments = review_section.find_elements(By.CSS_SELECTOR, "div.text-md.whitespace-pre-wrap.break-words.leading-relaxed.text-gray-900")
            for comment in comments:
                try:
                    span_element = comment.find_element(By.TAG_NAME, "span")
                except:
                    span_element = None
                
                if span_element:
                    driver.execute_script("arguments[0].click();", span_element)
                    time.sleep(1)  # Chờ một chút để nội dung hiển thị
                
                comment_text = comment.text
                comment_text = re.sub(r'Thu gọn$', '', comment_text, flags=re.IGNORECASE).strip()
                
                dict_review['comment'].append(comment_text)

            # Thêm thông tin URL và tiêu đề đánh giá vào từ điển
            length = len(dict_review['author'])
            dict_review['url'] = [review_url] * length
            dict_review['title'] = [review_title] * length

            # print(dict_review)
        except TimeoutException:
            print("Timeout! Thử lại...")
            driver.refresh()
        total_data.append(dict_review)
    
    return total_data
    
def save_to_csv(data: list[dict], filename: str):
    df = pd.DataFrame()
    
    for dict_review in data:
        try:
            df_temp = pd.DataFrame(dict_review)
            df = pd.concat([df, df_temp], ignore_index=True)
        except Exception as e:
            print(f"Lỗi khi chuyển đổi từ dictionary thành DataFrame: {e}")
            print(f"Dictionary gây ra lỗi có title là: {dict_review['title']}")
            continue
    
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
if __name__ == "__main__":
    driver = init_driver()

    base_url = "https://www.momo.vn/cinema/review"

    specific_urls_list = get_film_review_url(driver, base_url)
    print(f"Hoàn tất lấy các liên kết cần thiết. Tổng số liên kết đánh giá lấy được: {len(specific_urls_list)}")

    with open("list_of_dict_reviews.json", "w", encoding="utf-8-sig") as f:
        json.dump(specific_urls_list, f, ensure_ascii=False, indent=4)
    print("Dữ liệu đã được lưu vào list_of_dict_reviews.json")

    total_data = get_specific_film_review(driver, specific_urls_list)
    print("Hoàn tất lấy dữ liệu đánh giá phim.")

    # with open("film_reviews.json", "w", encoding="utf-8-sig") as f:
    #     json.dump(total_data, f, ensure_ascii=False, indent=4)
    # print("Dữ liệu đã được lưu vào film_reviews.json")

    save_to_csv(total_data, "film_reviews.csv")
    print("Dữ liệu đã được lưu vào film_reviews.csv")

    driver.quit()