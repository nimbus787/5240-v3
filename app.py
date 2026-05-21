import json
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from opencc import OpenCC
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)


# =========================================================
# MenuMate HK: UK Menu Allergy Assistant
# 功能：
# 1. 英文菜单图片 OCR
# 2. 英文菜名 / 菜单描述过敏原预测
# 3. 英文菜名翻译为繁体中文
# 4. 输出 14 类 UK / EU 主要过敏原风险提示
# =========================================================


# =========================
# 基础页面设置
# =========================

st.set_page_config(
    page_title="MenuMate HK",
    page_icon="🍽️",
    layout="wide"
)


# =========================
# Hugging Face 模型配置
# =========================

ALLERGEN_MODEL_REPO = "nimbus8858/menumate-distilbert-allergen-optimized"

OCR_MODEL_OPTIONS = {
    "Fast OCR - TrOCR Small Printed": "microsoft/trocr-small-printed",
    "Better OCR - TrOCR Base Printed": "microsoft/trocr-base-printed"
}

TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"


# =========================
# 14 类过敏原中文映射
# =========================

ALLERGEN_ZH = {
    "gluten": "麩質穀物",
    "crustaceans": "甲殼類",
    "eggs": "蛋類",
    "fish": "魚類",
    "peanuts": "花生",
    "soybeans": "大豆",
    "milk": "奶類 / 乳製品",
    "tree_nuts": "木本堅果",
    "celery": "芹菜",
    "mustard": "芥末",
    "sesame": "芝麻",
    "sulphites": "二氧化硫及亞硫酸鹽",
    "lupin": "羽扇豆",
    "molluscs": "軟體動物"
}


# =========================
# Keyword backup rules
# 说明：
# 这是模型预测之外的安全补充。
# 如果菜单中直接出现明确食材关键词，即使模型分数低，也可以提醒用户。
# =========================

KEYWORD_RULES = {
    "gluten": [
        "wheat", "flour", "bread", "breadcrumbs", "panko", "pasta",
        "spaghetti", "linguine", "noodle", "barley", "rye", "oat",
        "pastry", "batter", "bun", "cake", "cookie", "biscuit"
    ],
    "crustaceans": [
        "shrimp", "prawn", "prawns", "lobster", "crab", "crayfish", "scampi"
    ],
    "eggs": [
        "egg", "eggs", "omelette", "omelet", "mayonnaise", "mayo",
        "aioli", "hollandaise", "custard", "meringue", "quiche"
    ],
    "fish": [
        "fish", "salmon", "tuna", "cod", "anchovy", "sardine",
        "haddock", "sea bass", "trout", "mackerel", "fish sauce"
    ],
    "peanuts": [
        "peanut", "peanuts", "groundnut", "satay", "peanut butter"
    ],
    "soybeans": [
        "soy", "soya", "soybean", "tofu", "edamame", "miso",
        "tempeh", "soy sauce"
    ],
    "milk": [
        "milk", "cheese", "butter", "cream", "creamy", "yogurt",
        "yoghurt", "ghee", "buttermilk", "whey", "casein", "lactose",
        "parmesan", "mozzarella", "cheddar", "ricotta", "feta",
        "cream cheese", "sour cream", "custard", "ice cream"
    ],
    "tree_nuts": [
        "almond", "hazelnut", "walnut", "cashew", "pistachio",
        "pecan", "macadamia", "pine nut", "praline", "marzipan", "pesto"
    ],
    "celery": [
        "celery", "celeriac"
    ],
    "mustard": [
        "mustard", "dijon", "mustard seed", "mustard powder"
    ],
    "sesame": [
        "sesame", "sesame seed", "sesame oil", "tahini"
    ],
    "sulphites": [
        "sulphite", "sulphites", "sulfite", "sulfites",
        "wine", "red wine", "white wine", "dried fruit", "raisin",
        "dried apricot", "molasses"
    ],
    "lupin": [
        "lupin", "lupini", "lupin flour"
    ],
    "molluscs": [
        "oyster", "mussel", "clam", "squid", "octopus",
        "scallop", "calamari"
    ]
}


# =========================
# Hidden allergen warning rules
# 说明：
# 有些菜名不直接写过敏原，但烹饪方式可能暗示隐藏风险。
# 这些不是确定标签，而是 caution warnings。
# =========================

HIDDEN_RISK_RULES = {
    "stuffed": ["gluten", "eggs", "milk"],
    "battered": ["gluten", "eggs"],
    "breaded": ["gluten", "eggs"],
    "crispy": ["gluten"],
    "creamy": ["milk"],
    "cream sauce": ["milk"],
    "dressing": ["mustard", "eggs", "milk"],
    "sauce": ["milk", "mustard", "sulphites"],
    "pastry": ["gluten", "milk", "eggs"],
    "pie": ["gluten", "milk", "eggs"],
    "tartar sauce": ["eggs"],
    "hollandaise": ["eggs", "milk"],
    "pesto": ["tree_nuts", "milk"],
    "burger": ["gluten", "sesame"],
    "bun": ["gluten", "sesame"]
}


# =========================
# Food glossary
# 说明：
# 通用翻译模型对菜单词汇不稳定，所以用 glossary 修正常见菜名词汇。
# =========================

FOOD_GLOSSARY = {
    "pork": "豬肉",
    "british pork": "英式豬肉",
    "stuffed": "釀",
    "sage": "鼠尾草",
    "lemon": "檸檬",
    "garlic": "蒜香",
    "fish and chips": "炸魚薯條",
    "prawn": "大蝦",
    "shrimp": "蝦",
    "linguine": "扁意粉",
    "carbonara": "卡邦尼意粉",
    "caesar salad": "凱撒沙律",
    "mushroom soup": "蘑菇湯",
    "cheesecake": "芝士蛋糕",
    "chicken": "雞肉",
    "salmon": "三文魚",
    "beef": "牛肉",
    "burger": "漢堡",
    "cream": "忌廉",
    "cheese": "芝士",
    "butter": "牛油",
    "almond": "杏仁",
    "sesame": "芝麻",
    "mustard": "芥末"
}


# =========================
# 工具函数
# =========================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize_text(text: str) -> str:
    text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    # 如果 OCR 输出全大写，转成更适合翻译和模型理解的形式
    if text.isupper():
        text = text.title()

    return text


def normalize_for_matching(text: str) -> str:
    text = str(text).lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def keyword_match(text: str, keyword: str) -> bool:
    text = normalize_for_matching(text)
    keyword = normalize_for_matching(keyword)

    if not keyword:
        return False

    pattern = r"(?<![a-z0-9])" + re.escape(keyword).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def split_menu_lines(raw_text: str) -> List[str]:
    if not raw_text:
        return []

    text = raw_text.replace("\r", "\n")
    text = re.sub(r"[•|]", "\n", text)

    candidate_lines = []

    for line in text.split("\n"):
        line = normalize_text(line)

        # 删除常见价格格式
        line = re.sub(r"£\s?\d+(\.\d{1,2})?", "", line)
        line = re.sub(r"\bGBP\s?\d+(\.\d{1,2})?\b", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\b\d{1,3}\.\d{2}\b", "", line)

        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        lowered = line.lower()
        excluded_titles = {
            "starter", "starters", "main", "mains", "main courses",
            "dessert", "desserts", "drink", "drinks", "beverage",
            "beverages", "price", "menu", "restaurant"
        }

        if lowered in excluded_titles:
            continue

        if len(line) < 3 or line.isdigit():
            continue

        candidate_lines.append(line)

    unique_items = []
    seen = set()

    for item in candidate_lines:
        key = item.lower()
        if key not in seen:
            unique_items.append(item)
            seen.add(key)

    return unique_items


# =========================
# 加载模型
# =========================

@st.cache_resource(show_spinner=False)
def load_allergen_model():
    tokenizer = AutoTokenizer.from_pretrained(ALLERGEN_MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(ALLERGEN_MODEL_REPO)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    label_file = hf_hub_download(
        repo_id=ALLERGEN_MODEL_REPO,
        filename="allergen_labels.json",
        repo_type="model"
    )

    # 优先读取 F2 threshold 文件
    try:
        threshold_file = hf_hub_download(
            repo_id=ALLERGEN_MODEL_REPO,
            filename="allergen_thresholds_f2.json",
            repo_type="model"
        )
    except Exception:
        threshold_file = hf_hub_download(
            repo_id=ALLERGEN_MODEL_REPO,
            filename="allergen_thresholds.json",
            repo_type="model"
        )

    with open(label_file, "r", encoding="utf-8") as f:
        labels = json.load(f)

    with open(threshold_file, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    return tokenizer, model, labels, thresholds, device


@st.cache_resource(show_spinner=False)
def load_ocr_pipeline(model_name: str):
    device_id = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="image-to-text",
        model=model_name,
        device=device_id
    )


@st.cache_resource(show_spinner=False)
def load_translation_pipeline():
    device_id = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="translation",
        model=TRANSLATION_MODEL_NAME,
        tokenizer=TRANSLATION_MODEL_NAME,
        device=device_id
    )


@st.cache_resource(show_spinner=False)
def load_opencc_converter():
    return OpenCC("s2t")


# =========================
# OCR / Translation
# =========================

def preprocess_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    image = ImageOps.autocontrast(image)
    return image


def run_ocr(image: Image.Image, ocr_model_name: str) -> str:
    ocr_pipe = load_ocr_pipeline(ocr_model_name)
    processed_image = preprocess_image(image)

    result = ocr_pipe(processed_image)

    if isinstance(result, list) and len(result) > 0:
        return normalize_text(result[0].get("generated_text", ""))

    return ""


def glossary_translate(text: str) -> str:
    lower_text = text.lower().strip()

    if lower_text in FOOD_GLOSSARY:
        return FOOD_GLOSSARY[lower_text]

    # 对常见短语做简单组合翻译
    translated = text

    for english_term, chinese_term in sorted(FOOD_GLOSSARY.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = re.compile(re.escape(english_term), re.IGNORECASE)
        translated = pattern.sub(chinese_term, translated)

    if translated != text:
        return translated

    return ""


def translate_to_traditional_chinese(text: str, enable_translation: bool) -> str:
    if not enable_translation:
        return ""

    text = normalize_text(text)

    glossary_result = glossary_translate(text)

    if glossary_result:
        return glossary_result

    try:
        translator = load_translation_pipeline()
        converter = load_opencc_converter()

        result = translator(text, max_length=128)
        simplified_text = result[0]["translation_text"]
        traditional_text = converter.convert(simplified_text)

        return traditional_text
    except Exception:
        return "翻譯失敗"


# =========================
# Allergen prediction
# =========================

def predict_model_allergens(text: str) -> pd.DataFrame:
    tokenizer, model, labels, thresholds, device = load_allergen_model()

    text = normalize_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # =========================
    # 关键修复：
    # 只保留当前模型 forward() 支持的输入字段
    # 例如 DistilBERT 通常不接受 token_type_ids
    # =========================
    import inspect

    valid_forward_args = inspect.signature(model.forward).parameters

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if key in valid_forward_args
    }

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()[0]

    probabilities = sigmoid(logits)

    rows = []

    for label, prob in zip(labels, probabilities):
        threshold = thresholds.get(label, 0.5)
        predicted = int(prob >= threshold)

        rows.append({
            "label": label,
            "label_zh": ALLERGEN_ZH.get(label, label),
            "probability": float(prob),
            "threshold": float(threshold),
            "model_predicted": predicted
        })

    return pd.DataFrame(rows).sort_values(by="probability", ascending=False)
    tokenizer, model, labels, thresholds, device = load_allergen_model()

    text = normalize_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()[0]

    probabilities = sigmoid(logits)

    rows = []

    for label, prob in zip(labels, probabilities):
        threshold = thresholds.get(label, 0.5)
        predicted = int(prob >= threshold)

        rows.append({
            "label": label,
            "label_zh": ALLERGEN_ZH.get(label, label),
            "probability": float(prob),
            "threshold": float(threshold),
            "model_predicted": predicted
        })

    return pd.DataFrame(rows).sort_values(by="probability", ascending=False)


def detect_keyword_allergens(text: str) -> List[str]:
    detected_labels = []

    for label, keywords in KEYWORD_RULES.items():
        if any(keyword_match(text, keyword) for keyword in keywords):
            detected_labels.append(label)

    return detected_labels


def detect_hidden_risks(text: str) -> List[str]:
    hidden_labels = []

    for trigger, labels in HIDDEN_RISK_RULES.items():
        if keyword_match(text, trigger):
            hidden_labels.extend(labels)

    return sorted(list(set(hidden_labels)))


def assign_risk_level(final_labels: List[str], hidden_labels: List[str]) -> str:
    if not final_labels and not hidden_labels:
        return "Low / 未檢測到明顯風險"

    high_risk_labels = {"peanuts", "tree_nuts", "crustaceans", "molluscs", "fish"}

    if any(label in high_risk_labels for label in final_labels):
        return "High / 高風險"

    if len(final_labels) >= 2:
        return "High / 高風險"

    if hidden_labels:
        return "Medium / 中等風險"

    return "Medium / 中等風險"


def analyze_single_dish(text: str, enable_translation: bool) -> Dict[str, object]:
    text = normalize_text(text)

    model_df = predict_model_allergens(text)
    model_labels = model_df[model_df["model_predicted"] == 1]["label"].tolist()

    keyword_labels = detect_keyword_allergens(text)
    hidden_labels = detect_hidden_risks(text)

    final_labels = sorted(list(set(model_labels + keyword_labels)))
    risk_level = assign_risk_level(final_labels, hidden_labels)

    dish_zh = translate_to_traditional_chinese(text, enable_translation)

    final_labels_zh = [ALLERGEN_ZH.get(label, label) for label in final_labels]
    hidden_labels_zh = [ALLERGEN_ZH.get(label, label) for label in hidden_labels]

    top_model_signals = model_df.head(5).copy()
    top_model_signals["probability"] = top_model_signals["probability"].round(4)

    return {
        "English Dish Name": text,
        "繁體中文菜名": dish_zh,
        "Detected Allergens": ", ".join(final_labels) if final_labels else "No obvious allergen detected",
        "中文過敏原提示": "、".join(final_labels_zh) if final_labels_zh else "未檢測到明顯過敏原",
        "Hidden Risk Warning": "、".join(hidden_labels_zh) if hidden_labels_zh else "-",
        "Risk Level": risk_level,
        "Top Model Signals": top_model_signals
    }


def analyze_menu_items(menu_items: List[str], enable_translation: bool) -> pd.DataFrame:
    rows = []

    for item in menu_items:
        result = analyze_single_dish(item, enable_translation)
        rows.append({
            "English Dish Name": result["English Dish Name"],
            "繁體中文菜名": result["繁體中文菜名"],
            "Detected Allergens": result["Detected Allergens"],
            "中文過敏原提示": result["中文過敏原提示"],
            "Hidden Risk Warning": result["Hidden Risk Warning"],
            "Risk Level": result["Risk Level"]
        })

    return pd.DataFrame(rows)


# =========================
# Streamlit UI
# =========================

SAMPLE_MENU_TEXT = """Fish and chips with tartar sauce
Prawn linguine with garlic butter
Chicken Caesar salad
Mushroom soup with cream
British pork stuffed with sage, lemon and garlic
Cheesecake with almond crumble
Vegan tofu bowl with soy sauce
Shrimp cakes with panko, mayonnaise and dijon mustard"""


st.title("🍽️ MenuMate HK: UK Menu Allergy Assistant")

st.caption(
    "A menu-scanning assistant for Hong Kong residents traveling in the UK. "
    "The app uses OCR, a fine-tuned DistilBERT allergen classifier, and English-to-Traditional Chinese translation."
)

with st.sidebar:
    st.header("Settings")

    selected_ocr_label = st.selectbox(
        "OCR model",
        options=list(OCR_MODEL_OPTIONS.keys()),
        index=0
    )

    selected_ocr_model = OCR_MODEL_OPTIONS[selected_ocr_label]

    enable_translation = st.checkbox(
        "Translate dish names into Traditional Chinese",
        value=True
    )

    st.divider()

    st.markdown("### Pipeline Design")
    st.markdown(
        """
        **Pipeline 1:** OCR image-to-text  
        **Pipeline 2:** Fine-tuned DistilBERT allergen classifier  
        **Pipeline 3:** English-to-Traditional Chinese translation  
        """
    )

    st.markdown("### Fine-tuned Model")
    st.code(ALLERGEN_MODEL_REPO)

    st.warning(
        "This app provides AI-assisted allergen risk screening only. "
        "Users should always confirm ingredients with restaurant staff."
    )


tab1, tab2 = st.tabs(["Analyze Menu Text", "Analyze Menu Image"])


with tab1:
    st.markdown("## Step 1: Paste or type English menu text")

    col_text, col_button = st.columns([3, 1])

    with col_text:
        raw_menu_text = st.text_area(
            "Enter one dish per line.",
            value=SAMPLE_MENU_TEXT,
            height=220
        )

    with col_button:
        st.markdown("### Quick Test")
        if st.button("Load sample menu"):
            raw_menu_text = SAMPLE_MENU_TEXT

    st.markdown("## Step 2: Analyze allergen risks")

    if st.button("Analyze Text Menu"):
        menu_items = split_menu_lines(raw_menu_text)

        if not menu_items:
            st.warning("No valid menu items found.")
        else:
            with st.spinner("Analyzing menu items with fine-tuned DistilBERT..."):
                result_df = analyze_menu_items(
                    menu_items=menu_items,
                    enable_translation=enable_translation
                )

            st.success(f"Analysis completed. {len(result_df)} menu item(s) processed.")
            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download results as CSV",
                data=csv_data,
                file_name="menumate_allergen_results.csv",
                mime="text/csv"
            )

            st.markdown("### High-risk warnings")

            high_risk_df = result_df[result_df["Risk Level"].str.contains("High", na=False)]

            if high_risk_df.empty:
                st.info("No high-risk item was detected.")
            else:
                for _, row in high_risk_df.iterrows():
                    st.warning(
                        f"⚠️ {row['English Dish Name']} / {row['繁體中文菜名']}："
                        f"可能含有 {row['中文過敏原提示']}"
                    )


with tab2:
    st.markdown("## Step 1: Upload an English menu image")

    uploaded_file = st.file_uploader(
        "For best OCR results, upload a clear cropped dish name or menu line.",
        type=["png", "jpg", "jpeg"]
    )

    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col_img, col_ocr = st.columns([1, 1])

        with col_img:
            st.image(image, caption="Uploaded menu image", use_container_width=True)

        with col_ocr:
            st.markdown("### OCR Result")

            if st.button("Run OCR"):
                with st.spinner("Running OCR model..."):
                    try:
                        ocr_text = run_ocr(image, selected_ocr_model)
                        st.session_state.ocr_text = ocr_text

                        if ocr_text:
                            st.success("OCR completed.")
                        else:
                            st.warning("OCR returned empty text. Try cropping the image more tightly.")
                    except Exception as e:
                        st.error(f"OCR failed: {e}")

            if st.session_state.ocr_text:
                st.info(st.session_state.ocr_text)

    st.markdown("## Step 2: Review OCR text")

    image_menu_text = st.text_area(
        "Edit OCR result if necessary.",
        value=st.session_state.ocr_text,
        height=180
    )

    st.markdown("## Step 3: Analyze allergen risks")

    if st.button("Analyze Image Menu"):
        menu_items = split_menu_lines(image_menu_text)

        if not menu_items:
            st.warning("No valid menu items found. Please check the OCR result.")
        else:
            with st.spinner("Analyzing menu items with fine-tuned DistilBERT..."):
                result_df = analyze_menu_items(
                    menu_items=menu_items,
                    enable_translation=enable_translation
                )

            st.success(f"Analysis completed. {len(result_df)} menu item(s) processed.")
            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download results as CSV",
                data=csv_data,
                file_name="menumate_image_allergen_results.csv",
                mime="text/csv"
            )


st.divider()

st.markdown("## Important Notice")
st.markdown(
    """
    MenuMate HK provides AI-assisted allergen risk screening for educational and demonstration purposes.  
    The system may miss hidden ingredients or restaurant-specific cooking methods.  
    Users with food allergies should always confirm ingredients directly with restaurant staff before ordering.
    """
)
