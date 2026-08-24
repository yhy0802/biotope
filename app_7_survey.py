import os
import io
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image

from transformers import pipeline
from streamlit_js_eval import get_geolocation


# =========================================================
# 0. 기본 설정
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

EXCEL_FILE = os.path.join(
    BASE_DIR,
    "비오톱 유형 분류표_윤혜연작업.xlsx"
)

RECORD_FILE = os.path.join(
    BASE_DIR,
    "비오톱_현장조사기록.csv"
)

PHOTO_DIR = os.path.join(
    BASE_DIR,
    "survey_photos"
)

MODEL_NAME = "openai/clip-vit-large-patch14"

os.makedirs(
    PHOTO_DIR,
    exist_ok=True
)

st.set_page_config(
    page_title="AI 비오톱 현장조사",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 1. 모바일 UI
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 680px;
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-bottom: 4rem;
    }

    .main-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        font-size: 0.93rem;
        color: #6b7280;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }

    .step-title {
        font-size: 1.18rem;
        font-weight: 750;
        margin-top: 0.6rem;
        margin-bottom: 0.5rem;
    }

    .result-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px;
        background: #fafafa;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .location-card {
        border: 1px solid #dfe5e8;
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 8px;
        background: #f8fafc;
    }

    .small-text {
        font-size: 0.85rem;
        color: #6b7280;
    }

    div.stButton > button {
        width: 100%;
        min-height: 46px;
        font-weight: 700;
    }

    div[data-testid="stFileUploader"] {
        margin-bottom: 0.5rem;
    }

    div[data-testid="stCameraInput"] {
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 2. 데이터
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_excel(
        EXCEL_FILE,
        dtype=str
    )

    df.columns = df.columns.str.strip()

    df = df.ffill()

    required_columns = [
        "비오톱 구분",
        "대분류",
        "대분류(분류코드)",
        "중분류",
        "중분류(분류코드)",
        "소분류",
        "소분류(분류코드)"
    ]

    df = df.dropna(
        subset=["소분류"]
    )

    return (
        df[required_columns]
        .drop_duplicates()
    )


try:

    df = load_data()

except Exception as e:

    st.error(
        "비오톱 분류표를 불러오지 못했습니다."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# 3. AI 모델
# =========================================================

@st.cache_resource
def load_model():

    return pipeline(
        task="zero-shot-image-classification",
        model=MODEL_NAME
    )


# =========================================================
# 4. AI 프롬프트
# =========================================================

LARGE_PROMPTS = {

    "주거지":
        "residential area with apartments houses and housing complexes",

    "상업 업무지":
        "commercial business district office buildings shops",

    "주상 혼합지":
        "mixed residential and commercial urban neighborhood",

    "공공 용도지":
        "public institutional area school hospital government building",

    "공업지":
        "industrial factories warehouses industrial buildings",

    "공급처리 시설지":
        "utility treatment plant energy waste infrastructure",

    "교통 시설지":
        "transportation infrastructure railway road station airport parking",

    "특수지":
        "special purpose developed or disturbed artificial land",

    "하천":
        "river stream creek flowing water channel",

    "호소 및 습지":
        "lake reservoir pond marsh wetland",

    "해안":
        "coastal shoreline beach tidal flat",

    "산림":
        "forest woodland dense trees",

    "초지":
        "grassland meadow herbaceous vegetation",

    "경작지":
        "agricultural farmland crop field orchard",

    "조성녹지":
        "urban park landscaped green space",

    "나지 및 폐허지":
        "bare land vacant abandoned construction ground"
}


SPECIAL_PROMPTS = {

    "철도":
        "railway tracks steel rails railroad corridor",

    "철도시설":
        "railway tracks train station railroad infrastructure",

    "철도 관련시설":
        "railway tracks station railroad infrastructure",

    "철도역":
        "railway station train platforms tracks",

    "도로":
        "road highway street automobile transportation",

    "도로시설":
        "road highway street transportation infrastructure",

    "공항":
        "airport runway aircraft terminal aviation facility",

    "주차장":
        "parking lot parked cars parking spaces",

    "자연하천":
        "natural river stream natural banks vegetation",

    "인공하천":
        "engineered stream concrete banks",

    "소하천":
        "small natural stream creek",

    "모래톱":
        "sand bar inside river",

    "자갈톱":
        "gravel bar stones inside river",

    "저수지":
        "reservoir artificial lake permanent water",

    "저류지":
        "stormwater detention basin temporary retention pond",

    "습지":
        "wetland marsh shallow water wetland vegetation",

    "자연림":
        "natural forest irregular naturally growing trees",

    "인공림":
        "planted forest regularly spaced trees",

    "자연-인공림":
        "mixed natural and planted forest",

    "활엽수림":
        "broad leaved forest",

    "침엽수림":
        "coniferous forest",

    "혼효림":
        "mixed broad leaved and coniferous forest",

    "관목식생지":
        "shrubland woody shrubs",

    "하반림":
        "riparian forest riverbank forest",

    "논":
        "rice paddy field",

    "밭":
        "dry cultivated crop field",

    "과수원":
        "orchard planted fruit trees",

    "비닐하우스":
        "plastic agricultural greenhouse",

    "시설재배지":
        "greenhouse agricultural cultivation",

    "공원":
        "urban public park trees lawn walking paths",

    "근린공원":
        "neighborhood park trees lawn recreation",

    "완충녹지":
        "linear buffer green space",

    "경관녹지":
        "landscaped scenic green space",

    "수목원":
        "arboretum planted tree collection",

    "식물원":
        "botanical garden plant collection"
}


KEYWORD_PROMPTS = {

    "철도": "railway tracks railroad",

    "도로": "road highway",

    "공항": "airport runway",

    "하천": "river stream",

    "수로": "water channel",

    "습지": "wetland marsh",

    "산림": "forest woodland",

    "숲": "forest woodland",

    "초지": "grassland",

    "경작": "agricultural land",

    "농경": "farmland",

    "공원": "urban park",

    "녹지": "green space",

    "나지": "bare land",

    "아파트": "apartment residential",

    "주택": "housing",

    "학교": "school campus",

    "산업": "industrial area",

    "공장": "factory"
}


# =========================================================
# 5. AI 함수
# =========================================================

def make_prompt(label):

    label = str(label).strip()

    if label in SPECIAL_PROMPTS:
        return SPECIAL_PROMPTS[label]

    pieces = []

    for keyword, prompt in KEYWORD_PROMPTS.items():

        if keyword in label:
            pieces.append(prompt)

    if pieces:

        return (
            ", ".join(
                dict.fromkeys(pieces)
            )
            +
            f", landscape habitat category {label}"
        )

    return (
        f"landscape habitat type corresponding to "
        f"Korean biotope category {label}"
    )


def classify_candidates(
    classifier,
    image,
    candidates,
    custom_prompts=None
):

    prompts = []

    mapping = {}

    for candidate in candidates:

        if (
            custom_prompts
            and candidate in custom_prompts
        ):

            prompt = custom_prompts[candidate]

        else:

            prompt = make_prompt(candidate)

        prompts.append(prompt)

        mapping[prompt] = candidate


    results = classifier(
        image,
        candidate_labels=prompts
    )


    converted = []

    for result in results:

        converted.append(
            {
                "label":
                    mapping.get(
                        result["label"],
                        result["label"]
                    ),

                "score":
                    result["score"]
            }
        )

    return converted


def show_ai_results(results):

    rows = []

    for i, result in enumerate(
        results[:3]
    ):

        rows.append(
            {
                "순위": i + 1,

                "AI 추천": result["label"],

                "참고점수":
                    f"{result['score'] * 100:.1f}%"
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True
    )


# =========================================================
# 6. 상단
# =========================================================

st.markdown(
    '<div class="main-title">🌿 AI 비오톱 현장조사</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-subtitle">
    휴대폰으로 현재 위치와 현장사진을 기록하고,
    AI 추천을 참고하여 조사자가 비오톱 유형을 최종 확인합니다.
    </div>
    """,
    unsafe_allow_html=True
)


tab_survey, tab_records = st.tabs(
    [
        "📝 조사하기",
        "📚 조사기록"
    ]
)


# =========================================================
# TAB 1
# =========================================================

with tab_survey:

    # -----------------------------------------------------
    # STEP 1 조사정보
    # -----------------------------------------------------

    st.markdown(
        '<div class="step-title">① 조사정보</div>',
        unsafe_allow_html=True
    )

    investigator = st.text_input(
        "조사자",
        placeholder="이름 또는 조사자 코드"
    )

    site_name = st.text_input(
        "조사지점",
        placeholder="예: ○○시 ○○공원"
    )


    # -----------------------------------------------------
    # STEP 2 GPS
    # -----------------------------------------------------

    st.markdown(
        '<div class="step-title">② 현재 위치</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "아래 위치 버튼을 누르고 브라우저의 위치정보 사용을 허용하세요."
    )


    location = get_geolocation()


    if location:

        # ---------------------------------------------
        # 오류
        # ---------------------------------------------

        if "error" in location:

            error_info = location["error"]

            error_code = error_info.get(
                "code",
                0
            )

            error_message = error_info.get(
                "message",
                ""
            )


            if error_code == 1:

                st.error(
                    "위치정보 권한이 거부되었습니다. "
                    "브라우저 설정에서 위치 권한을 허용해주세요."
                )

            else:

                st.warning(
                    f"위치를 가져오지 못했습니다: "
                    f"{error_message}"
                )


        # ---------------------------------------------
        # 정상
        # ---------------------------------------------

        elif "coords" in location:

            coords = location["coords"]

            latitude = coords.get(
                "latitude"
            )

            longitude = coords.get(
                "longitude"
            )

            accuracy = coords.get(
                "accuracy"
            )


            st.session_state[
                "latitude"
            ] = latitude

            st.session_state[
                "longitude"
            ] = longitude

            st.session_state[
                "accuracy"
            ] = accuracy


    latitude = st.session_state.get(
        "latitude"
    )

    longitude = st.session_state.get(
        "longitude"
    )

    accuracy = st.session_state.get(
        "accuracy"
    )


    if (
        latitude is not None
        and longitude is not None
    ):

        accuracy_text = (
            f"{accuracy:.1f} m"
            if isinstance(
                accuracy,
                (int, float)
            )
            else "-"
        )

        st.markdown(
            f"""
            <div class="location-card">

            <b>📍 현재 위치 확인 완료</b>

            <br><br>

            위도: {latitude:.6f}<br>
            경도: {longitude:.6f}<br>

            <span class="small-text">
            GPS 정확도: 약 {accuracy_text}
            </span>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "아직 위치정보를 가져오지 않았습니다."
        )


    # -----------------------------------------------------
    # STEP 3 사진
    # -----------------------------------------------------

    st.markdown(
        '<div class="step-title">③ 현장사진</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "분류 대상뿐 아니라 주변 토지이용과 식생이 "
        "함께 보이도록 촬영하면 AI 분석에 도움이 됩니다."
    )


    camera_photo = st.camera_input(
        "📷 현장사진 촬영"
    )


    if camera_photo is not None:

        photo_bytes = (
            camera_photo.getvalue()
        )

        st.session_state[
            "survey_photo_bytes"
        ] = photo_bytes


        image = Image.open(
            io.BytesIO(photo_bytes)
        ).convert("RGB")


        st.session_state[
            "survey_image"
        ] = image


        st.image(
            image,
            caption="촬영한 현장사진",
            use_container_width=True
        )


        if st.button(
            "🤖 AI 분석 시작",
            type="primary"
        ):

            with st.spinner(
                "AI 모델을 불러오고 있습니다..."
            ):

                classifier = load_model()


            large_candidates = sorted(
                df["대분류"]
                .drop_duplicates()
                .tolist()
            )


            with st.spinner(
                "대분류를 분석하고 있습니다..."
            ):

                results = classify_candidates(
                    classifier,
                    image,
                    large_candidates,
                    custom_prompts=LARGE_PROMPTS
                )


            st.session_state[
                "large_results"
            ] = results

            st.session_state.pop(
                "middle_results",
                None
            )

            st.session_state.pop(
                "small_results",
                None
            )


    # -----------------------------------------------------
    # STEP 4 대분류
    # -----------------------------------------------------

    if "large_results" in st.session_state:

        st.markdown(
            '<div class="step-title">④ 대분류 확인</div>',
            unsafe_allow_html=True
        )

        large_results = st.session_state[
            "large_results"
        ]

        show_ai_results(
            large_results
        )


        large_options = sorted(
            df["대분류"]
            .drop_duplicates()
            .tolist()
        )


        ai_large = (
            large_results[0]["label"]
        )


        large_index = (
            large_options.index(ai_large)
            if ai_large in large_options
            else 0
        )


        selected_large = st.selectbox(
            "조사자 최종 대분류",
            large_options,
            index=large_index
        )


        if st.button(
            "다음: 중분류 분석"
        ):

            middle_df = df[
                df["대분류"]
                ==
                selected_large
            ]

            candidates = sorted(
                middle_df[
                    "중분류"
                ]
                .drop_duplicates()
                .tolist()
            )


            classifier = load_model()

            image = st.session_state[
                "survey_image"
            ]


            with st.spinner(
                "중분류를 분석하고 있습니다..."
            ):

                results = classify_candidates(
                    classifier,
                    image,
                    candidates
                )


            st.session_state[
                "middle_results"
            ] = results

            st.session_state[
                "confirmed_large"
            ] = selected_large

            st.session_state.pop(
                "small_results",
                None
            )


    # -----------------------------------------------------
    # STEP 5 중분류
    # -----------------------------------------------------

    if "middle_results" in st.session_state:

        st.markdown(
            '<div class="step-title">⑤ 중분류 확인</div>',
            unsafe_allow_html=True
        )


        middle_results = st.session_state[
            "middle_results"
        ]

        show_ai_results(
            middle_results
        )


        confirmed_large = st.session_state[
            "confirmed_large"
        ]


        middle_df = df[
            df["대분류"]
            ==
            confirmed_large
        ]


        middle_options = sorted(
            middle_df[
                "중분류"
            ]
            .drop_duplicates()
            .tolist()
        )


        ai_middle = (
            middle_results[0]["label"]
        )


        middle_index = (
            middle_options.index(ai_middle)
            if ai_middle in middle_options
            else 0
        )


        selected_middle = st.selectbox(
            "조사자 최종 중분류",
            middle_options,
            index=middle_index
        )


        if st.button(
            "다음: 소분류 분석"
        ):

            small_df = middle_df[
                middle_df["중분류"]
                ==
                selected_middle
            ]


            candidates = sorted(
                small_df[
                    "소분류"
                ]
                .drop_duplicates()
                .tolist()
            )


            classifier = load_model()

            image = st.session_state[
                "survey_image"
            ]


            with st.spinner(
                "소분류를 분석하고 있습니다..."
            ):

                results = classify_candidates(
                    classifier,
                    image,
                    candidates
                )


            st.session_state[
                "small_results"
            ] = results

            st.session_state[
                "confirmed_middle"
            ] = selected_middle


    # -----------------------------------------------------
    # STEP 6 소분류 및 최종저장
    # -----------------------------------------------------

    if "small_results" in st.session_state:

        st.markdown(
            '<div class="step-title">⑥ 최종 판정</div>',
            unsafe_allow_html=True
        )


        small_results = st.session_state[
            "small_results"
        ]


        show_ai_results(
            small_results
        )


        confirmed_large = st.session_state[
            "confirmed_large"
        ]

        confirmed_middle = st.session_state[
            "confirmed_middle"
        ]


        final_df = df[
            (df["대분류"] == confirmed_large)
            &
            (df["중분류"] == confirmed_middle)
        ]


        small_options = sorted(
            final_df[
                "소분류"
            ]
            .drop_duplicates()
            .tolist()
        )


        ai_small = (
            small_results[0]["label"]
        )


        small_index = (
            small_options.index(ai_small)
            if ai_small in small_options
            else 0
        )


        selected_small = st.selectbox(
            "조사자 최종 소분류",
            small_options,
            index=small_index
        )


        selected_row = final_df[
            final_df["소분류"]
            ==
            selected_small
        ].iloc[0]


        st.markdown(
            f"""
            <div class="result-card">

            <b>최종 비오톱 유형</b>

            <br><br>

            {selected_row['대분류']}
            →
            {selected_row['중분류']}
            →
            <b>{selected_row['소분류']}</b>

            <br><br>

            코드:
            {selected_row['소분류(분류코드)']}

            </div>
            """,
            unsafe_allow_html=True
        )


        memo = st.text_area(
            "현장메모",
            placeholder=(
                "식생, 토지이용, 주변환경, "
                "특이사항 등을 기록하세요."
            ),
            height=110
        )


        if st.button(
            "💾 조사결과 저장",
            type="primary"
        ):

            if (
                latitude is None
                or longitude is None
            ):

                st.error(
                    "현재 위치를 먼저 확인해주세요."
                )

                st.stop()


            now = datetime.now()

            timestamp = now.strftime(
                "%Y%m%d_%H%M%S"
            )


            # =============================================
            # 사진 저장
            # =============================================

            photo_filename = ""


            if (
                "survey_photo_bytes"
                in st.session_state
            ):

                photo_filename = (
                    f"survey_{timestamp}.jpg"
                )

                photo_path = os.path.join(
                    PHOTO_DIR,
                    photo_filename
                )


                saved_image = Image.open(
                    io.BytesIO(
                        st.session_state[
                            "survey_photo_bytes"
                        ]
                    )
                ).convert("RGB")


                saved_image.save(
                    photo_path,
                    quality=90
                )


            # =============================================
            # AI 결과
            # =============================================

            ai_large = (
                st.session_state[
                    "large_results"
                ][0]["label"]
            )

            ai_middle = (
                st.session_state[
                    "middle_results"
                ][0]["label"]
            )

            ai_small = (
                st.session_state[
                    "small_results"
                ][0]["label"]
            )


            record = {

                "조사일시":
                    now.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "조사자":
                    investigator,

                "조사지점":
                    site_name,

                "위도":
                    latitude,

                "경도":
                    longitude,

                "GPS정확도_m":
                    accuracy,

                "AI추천_대분류":
                    ai_large,

                "AI추천_중분류":
                    ai_middle,

                "AI추천_소분류":
                    ai_small,

                "최종_비오톱구분":
                    selected_row[
                        "비오톱 구분"
                    ],

                "최종_대분류":
                    selected_row[
                        "대분류"
                    ],

                "대분류코드":
                    selected_row[
                        "대분류(분류코드)"
                    ],

                "최종_중분류":
                    selected_row[
                        "중분류"
                    ],

                "중분류코드":
                    selected_row[
                        "중분류(분류코드)"
                    ],

                "최종_소분류":
                    selected_row[
                        "소분류"
                    ],

                "소분류코드":
                    selected_row[
                        "소분류(분류코드)"
                    ],

                "현장메모":
                    memo,

                "사진파일":
                    photo_filename
            }


            new_record = pd.DataFrame(
                [record]
            )


            if os.path.exists(
                RECORD_FILE
            ):

                old_records = pd.read_csv(
                    RECORD_FILE,
                    dtype=str
                )

                save_df = pd.concat(
                    [
                        old_records,
                        new_record
                    ],
                    ignore_index=True
                )

            else:

                save_df = new_record


            save_df.to_csv(
                RECORD_FILE,
                index=False,
                encoding="utf-8-sig"
            )


            st.success(
                "✅ 현장조사 결과가 저장되었습니다."
            )


# =========================================================
# TAB 2
# =========================================================

with tab_records:

    st.subheader(
        "📚 조사기록"
    )


    if not os.path.exists(
        RECORD_FILE
    ):

        st.info(
            "아직 저장된 조사기록이 없습니다."
        )

    else:

        records = pd.read_csv(
            RECORD_FILE,
            dtype=str
        )


        st.metric(
            "누적 조사건수",
            len(records)
        )


        display_columns = [
            "조사일시",
            "조사자",
            "조사지점",
            "최종_대분류",
            "최종_중분류",
            "최종_소분류"
        ]


        available_columns = [
            col
            for col in display_columns
            if col in records.columns
        ]


        st.dataframe(
            records[
                available_columns
            ],
            hide_index=True,
            use_container_width=True
        )


        csv_data = records.to_csv(
            index=False
        )


        st.download_button(
            "📥 전체 조사기록 내려받기",
            data=csv_data.encode(
                "utf-8-sig"
            ),
            file_name=(
                "비오톱_현장조사기록.csv"
            ),
            mime="text/csv"
        )


# =========================================================
# 하단
# =========================================================

st.write("")
st.caption(
    "※ AI 분석은 현장조사자의 판정을 지원하기 위한 참고정보입니다."
)