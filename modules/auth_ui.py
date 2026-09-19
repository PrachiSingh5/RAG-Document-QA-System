import streamlit as st

from modules.auth import login, signup


def render_login_page():
    """
    VERSION 1 -- the original login/signup screen: simple tabs,
    centered form, no custom visual theme. The most reliable/plain
    of the three versions.
    """

    left, center, right = st.columns([1, 1.4, 1])

    with center:

        st.markdown(
            "<h1 style='text-align:center;'>📄 AI PDF Assistant</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align:center; color: #6b7280;'>"
            "Log in or create an account to continue.</p>",
            unsafe_allow_html=True
        )

        tab_login, tab_signup = st.tabs(["🔑 Login", "🆕 Sign Up"])

        with tab_login:

            with st.form("login_form"):

                login_email = st.text_input("Email")
                login_password = st.text_input("Password", type="password")
                login_submitted = st.form_submit_button(
                    "Login", use_container_width=True
                )

            if login_submitted:

                success, message, user = login(login_email, login_password)

                if success:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error(f"❌ {message}")

        with tab_signup:

            with st.form("signup_form"):

                signup_name = st.text_input("Full Name")
                signup_email = st.text_input("Email", key="signup_email")
                signup_password = st.text_input(
                    "Password", type="password", key="signup_pw"
                )
                signup_confirm = st.text_input(
                    "Confirm Password", type="password", key="signup_confirm"
                )
                signup_submitted = st.form_submit_button(
                    "Create Account", use_container_width=True
                )

            if signup_submitted:

                success, message, user = signup(
                    signup_name, signup_email, signup_password, signup_confirm
                )

                if success:
                    st.success(f"✅ {message} Please log in from the Login tab.")
                else:
                    st.error(f"❌ {message}")