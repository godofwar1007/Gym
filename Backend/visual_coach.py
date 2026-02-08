
import base64
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import GROQ_API_KEY

class VisualChain:
    def __init__(self):
        # 1. Vision Model (The Eyes)
        self.vlm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct", # Updated to latest Vision model on Groq
            temperature=0.0,
            api_key=GROQ_API_KEY
        )
        
        # 2. Text Model (The Coach Brain)
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
        
        # 3. Coach Prompt
        self.coach_prompt = ChatPromptTemplate.from_template(
            """
            You are an elite Sports Performance Coach.
            
            INPUT CONTEXT:
            - Exercise: {exercise_name}
            - Technical Analysis: "{observation}"
            
            TASK:
            Translate the technical analysis into a clear, helpful feedback summary.
            
            OUTPUT FORMAT (Strictly follow this layout):
            
            **Correction 1 (Priority):**
            * **The Error:** (Simple English. What are they doing wrong?)
            * **The Fix:** (Specific actionable step)
            * **The Cue:** (2-4 word mental trigger)
            
            **Correction 2 (Secondary):**
            (If no second error, write "None")
            * **The Error:** ...
            * **The Fix:** ...
            * **The Cue:** ...
            """
        )
        self.coach_chain = self.coach_prompt | self.llm | StrOutputParser()

    def analyze_images(self, user_img_b64, trainer_img_b64, exercise_name="Exercise"):
        """
        Takes Base64 images (with skeletons already drawn), gets technical analysis, then converts to coaching cue.
        """
        print(f"👀 VLM analyzing {exercise_name}...")
        
        vlm_messages = [
            SystemMessage(content="You are a Biomechanics Expert. Analyze the skeletal overlays in the images."),
            HumanMessage(content=[
                {
                    "type": "text", 
                    "text": f"""
                    TASK: Compare the Biomechanics of the USER (Image 1) vs. the TRAINER (Image 2).
                    
                    CONTEXT: 
                    These images have skeletal lines drawn on them (Green joints, Red lines). 
                    - Image 1: User performing {exercise_name}.
                    - Image 2: Trainer performing {exercise_name} (Perfect Form).

                    INSTRUCTIONS:
                    1. Compare the JOINT ANGLES based on the red lines.
                    2. Look specifically at: Spine Neutrality, Knee Tracking, Hip Depth, Arm Alignment.
                    3. IGNORE background. Focus ONLY on the geometry of the skeleton.

                    OUTPUT:
                    Identify the geometric differences. Be technical (e.g., "User's knee valgus angle is greater").
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{user_img_b64}"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{trainer_img_b64}"}
                }
                
            ])
        ]
        
        try:
            # Get the technical description
            vlm_response = self.vlm.invoke(vlm_messages)
            technical_observation = vlm_response.content
            
            # Generate Cue
            final_cue = self.coach_chain.invoke({
                "observation": technical_observation,
                "exercise_name": exercise_name
            })
            
            return final_cue
        except Exception as e:
            print(f"Error in VLM chain: {e}")
            return "Could not generate feedback. Please try again."

    def chat_with_coach(self, message, history_context=[]):
        """
        Chat with the coach, allowing for follow-up questions based on history.
        """
        chat_prompt = ChatPromptTemplate.from_template(
            """
            You are "Coach Maverick", an elite, high-energy Sports Performance Coach.
            
            YOUR CONTEXT (User's Recent Training History):
            {history}
            
            CURRENT USER MESSAGE:
            "{input}"
            
            INSTRUCTIONS:
            1. Answer the user's question directly.
            2. If they ask about their performance, REFER TO THE HISTORY.
            3. Be encouraging but technical (use terms like "valgus", "neutral spine", "hip drive").
            4. Keep answers concise (under 3 sentences unless detailed explanation is requested).
            5. Do NOT hallucinate. If history is empty, say you haven't seen them train yet.
            """
        )
        
        # Format history into a readable string
        history_str = "No recent history."
        if history_context:
            history_str = ""
            for session in history_context:
                history_str += f"- [Date: {session.get('timestamp')}] Exercise: {session.get('exercise_name')}, Score: {session.get('form_score')}, Errors: {session.get('mistakes')}\n"

        chain = chat_prompt | self.llm | StrOutputParser()
        
        try:
            return chain.invoke({
                "history": history_str,
                "input": message
            })
        except Exception as e:
            print(f"Error in Chat Chain: {e}")
            return "I'm having trouble connecting to the coaching server right now. Let's focus on your next set."
