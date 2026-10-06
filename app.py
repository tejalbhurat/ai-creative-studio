from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from creative_pipeline import run_creative_pipeline


st.set_page_config(page_title="AI Creative Studio", page_icon="🎨")
st.title("AI Creative Studio")
st.caption("Creative Director → image generation → Vision Critic")

user_request = st.text_area(
    "Describe the image you want...",
    placeholder="Create a premium dark Instagram campaign for an AI productivity product.",
    height=180,
)

if st.button("Generate Image", type="primary"):
    brief = user_request.strip()
    if not brief:
        st.warning("Enter a creative brief first.")
    else:
        try:
            with st.spinner("Creative agents are planning, generating, and evaluating..."):
                st.session_state["pipeline_result"] = run_creative_pipeline(brief)
        except Exception as error:
            st.error(f"Creative pipeline failed: {error}")

result = st.session_state.get("pipeline_result")
if result:
    st.subheader("Generated image")
    st.image(result["final_image"], caption="Generated image", use_container_width=True)

    with st.expander("View optimized image prompt"):
        st.write(result["final_prompt"])
        if result.get("text_content"):
            st.caption(f"Text integration: {result.get('text_content')}")

    for attempt in result["attempts"]:
        decision = attempt["critic"]["decision"] if attempt["critic"] else "not completed"
        st.write(
            f"Attempt {attempt['attempt']}/{3}: "
            f"{'accepted' if decision == 'ACCEPT' else decision}"
        )

    if result["status"] == "vision_rate_limited":
        st.warning(
            "Image generated successfully. Vision Critic is temporarily "
            "unavailable due to a rate limit, so automatic quality evaluation "
            "and regeneration were skipped."
        )
    elif result["critic_result"]:
        critic = result["critic_result"]
        st.subheader("AI critique")
        columns = st.columns(5)
        scores = [
            ("Overall", "overall_score"),
            ("Prompt adherence", "prompt_adherence"),
            ("Composition", "composition"),
            ("Visual quality", "visual_quality"),
            ("Brand fit", "brand_fit"),
        ]
        for column, (label, key) in zip(columns, scores):
            column.metric(label, f"{critic[key]}/10")

        with st.expander("Realism and typography evaluation"):
            st.write(
                f"Realism / naturalness: {critic['realism_naturalness']}/10"
            )
            st.write(f"AI artificiality: {critic['ai_artificiality']}/10")
            if result.get("text_content"):
                st.write(
                    f"Text integration: {critic['text_integration']}/10"
                )
                st.write(
                    f"Typography quality: {critic['typography_quality']}/10"
                )

        if critic["issues"]:
            st.write("**Issues**")
            for issue in critic["issues"]:
                st.write(f"- {issue}")
        if critic["improvements"]:
            st.write("**Improvements**")
            for improvement in critic["improvements"]:
                st.write(f"- {improvement}")
        if result["status"] == "accepted":
            st.success("Accepted by the Vision Critic.")
        else:
            st.warning("Maximum generation attempts reached.")
