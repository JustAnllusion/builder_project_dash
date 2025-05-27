import streamlit as st
import yaml
from yaml.loader import SafeLoader
import hashlib
from streamlit_cookies_manager import EncryptedCookieManager

cookies = EncryptedCookieManager(
    prefix="myapp_",
    password="supersecretpassword"  # обязательно поменяйте на свой надёжный пароль
)

if not cookies.ready():
    st.stop()

def load_config(config_path='config.yaml'):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=SafeLoader)
    return config

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def login():
    config = load_config()
    credentials = config.get('credentials', {}).get('usernames', {})

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if cookies.get("logged_in") == "0":
        username = cookies.get("username")
        name = cookies.get("name")
        
        if username in credentials:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.name = name
            return name, True, username
        else:
            # Если username из cookies не валиден, очищаем куки
            cookies.delete("logged_in")
            cookies.delete("username")
            cookies.delete("name")
            cookies.save()

    if st.session_state.logged_in:
        return st.session_state.name, True, st.session_state.username

    with st.form("login_form", clear_on_submit=True):
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Войти")

        if submitted:
            if username_input in credentials:
                user = credentials[username_input]
                stored_password = user.get("password")
                if stored_password == password_input or stored_password == hash_password(password_input):
                    st.session_state.logged_in = True
                    st.session_state.username = username_input
                    full_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()
                    st.session_state.name = full_name if full_name else username_input
                    
                    cookies["logged_in"] = "1"
                    cookies["username"] = username_input
                    cookies["name"] = st.session_state.name
                    cookies.save()
                    st.rerun()
            st.error("Неверное имя пользователя или пароль")

    return None, False, None

def logout():
    st.session_state.logged_in = False
    st.session_state.pop("username", None)
    st.session_state.pop("name", None)
    cookies.delete("logged_in")
    cookies.delete("username")
    cookies.delete("name")
    cookies.save()
    st.rerun()
