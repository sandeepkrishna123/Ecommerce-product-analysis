TITLE_STYLE = """
<style>
    .styled-title {
        font-family: 'Arial', sans-serif;
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(90deg, #008080, #00BFFF); /* Deep Teal to Sky Blue */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 10px;
        margin-bottom: 20px;
    }
</style>
"""

SIDEBAR_STYLE = """
<style>
    /* Sidebar title styling */
    .sidebar-title {
        font-family: 'Arial', sans-serif;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        margin-bottom: 20px;
        background: linear-gradient(90deg, #FF4500, #FFD700); /* Sunset Red to Golden Yellow */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Sidebar button styling */
    .stButton>button {
        width: 100%;
        text-align: center;
        padding: 12px;
        margin: 5px 0;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        background: linear-gradient(90deg, #90EE90, #20B2AA);
        color: white;
        border: none;
        transition: 0.3s;
    }

    /* Hover effect for buttons */
    .stButton>button:hover {
        background: linear-gradient(90deg, #20B2AA ,#90EE90);
        transform: scale(1.05);
    }

    /* Download button styling */
    .stDownloadButton>button {
        width: 100%;
        text-align: center;
        padding: 12px;
        margin: 5px 0;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        background: linear-gradient(90deg, #8A2BE2, #6A0DAD, #FF00FF); /* Violet to Purple to Magenta */
        color: white;
        border: none;
        transition: 0.3s;
    }

    /* Hover effect for download button */
    .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #FF00FF, #6A0DAD, #8A2BE2); /* Reverse gradient */
        transform: scale(1.05);
    }
</style>
"""
