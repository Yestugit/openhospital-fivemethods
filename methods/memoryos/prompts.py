"""Prompts adapted from BAI-LAB/MemoryOS, memoryos-pypi/prompts.py.

Profile analysis preserves the reference prompts and plain-text output.
Continuity, summaries and knowledge extraction use JSON response envelopes.
"""

CONTINUITY_SYSTEM = (
    "You are a conversation continuity detector. Decide whether the two pages "
    "are a true continuation without a topic shift. "
    'Return only JSON {"continuous": true} or {"continuous": false}.'
)
CONTINUITY_USER = """Check whether these two conversation pages are continuous.

Previous Page:
{previous}

Current Page:
{current}
"""

META_INFO_SYSTEM = """You are a conversation meta-summary updater. Your task is to:
1. Preserve relevant context from the previous meta-summary.
2. Integrate new information from the current dialogue.
3. Keep the updated summary concise, at most 1-2 sentences.
Return only JSON {"meta_info": "updated summary"}."""
META_INFO_USER = """Update the conversation meta-summary by incorporating the new dialogue while maintaining continuity.

Guidelines:
1. Start from the previous meta-summary when it exists.
2. Add or update information based on the new dialogue.
3. Maintain context coherence.

Previous Meta-summary: {previous_summary}
New Dialogue:
{conversation}
"""

TOPIC_SYSTEM = (
    "You are an expert in analyzing dialogue topics. Generate concise summaries. "
    "No more than two topics. Be as brief as possible. "
    'Return only JSON {"summaries": [{"theme": "brief theme", '
    '"keywords": ["keyword"], "content": "brief topic summary"}]}.'
)
TOPIC_USER = """Summarize the following dialogue into extremely concise subtopics, with a maximum of two themes.
Use just a few words for each theme and summary, and provide its keywords.

Conversation content:
{conversation}
"""

PROFILE_SYSTEM = """You are a professional user preference analysis assistant. Your task is to analyze the user's personality preferences from the given dialogue based on the provided dimensions.

For each dimension:
1. Carefully read the conversation and determine if the dimension is reflected.
2. If reflected, determine the user's preference level: High / Medium / Low, and briefly explain the reasoning, including time, people, and context if possible.
3. If the dimension is not reflected, do not extract or list it.

Focus only on the user's preferences and traits for the personality analysis section.
Output only the user profile section.
"""

PROFILE_USER = """Please analyze the latest user-AI conversation below and update the user profile based on the 90 personality preference dimensions.

Here are the 90 dimensions and their explanations:

[Psychological Model (Basic Needs & Personality)]
Extraversion: Preference for social activities.
Openness: Willingness to embrace new ideas and experiences.
Agreeableness: Tendency to be friendly and cooperative.
Conscientiousness: Responsibility and organizational ability.
Neuroticism: Emotional stability and sensitivity.
Physiological Needs: Concern for comfort and basic needs.
Need for Security: Emphasis on safety and stability.
Need for Belonging: Desire for group affiliation.
Need for Self-Esteem: Need for respect and recognition.
Cognitive Needs: Desire for knowledge and understanding.
Aesthetic Appreciation: Appreciation for beauty and art.
Self-Actualization: Pursuit of one's full potential.
Need for Order: Preference for cleanliness and organization.
Need for Autonomy: Preference for independent decision-making and action.
Need for Power: Desire to influence or control others.
Need for Achievement: Value placed on accomplishments.

[AI Alignment Dimensions]
Helpfulness: Whether the AI's response is practically useful to the user. (This reflects user's expectation of AI)
Honesty: Whether the AI's response is truthful. (This reflects user's expectation of AI)
Safety: Avoidance of sensitive or harmful content. (This reflects user's expectation of AI)
Instruction Compliance: Strict adherence to user instructions. (This reflects user's expectation of AI)
Truthfulness: Accuracy and authenticity of content. (This reflects user's expectation of AI)
Coherence: Clarity and logical consistency of expression. (This reflects user's expectation of AI)
Complexity: Preference for detailed and complex information.
Conciseness: Preference for brief and clear responses.

[Content Platform Interest Tags]
Science Interest: Interest in science topics.
Education Interest: Concern with education and learning.
Psychology Interest: Interest in psychology topics.
Family Concern: Interest in family and parenting.
Fashion Interest: Interest in fashion topics.
Art Interest: Engagement with or interest in art.
Health Concern: Concern with physical health and lifestyle.
Financial Management Interest: Interest in finance and budgeting.
Sports Interest: Interest in sports and physical activity.
Food Interest: Passion for cooking and cuisine.
Travel Interest: Interest in traveling and exploring new places.
Music Interest: Interest in music appreciation or creation.
Literature Interest: Interest in literature and reading.
Film Interest: Interest in movies and cinema.
Social Media Activity: Frequency and engagement with social media.
Tech Interest: Interest in technology and innovation.
Environmental Concern: Attention to environmental and sustainability issues.
History Interest: Interest in historical knowledge and topics.
Political Concern: Interest in political and social issues.
Religious Interest: Interest in religion and spirituality.
Gaming Interest: Enjoyment of video games or board games.
Animal Concern: Concern for animals or pets.
Emotional Expression: Preference for direct vs. restrained emotional expression.
Sense of Humor: Preference for humorous or serious communication style.
Information Density: Preference for detailed vs. concise information.
Language Style: Preference for formal vs. casual tone.
Practicality: Preference for practical advice vs. theoretical discussion.

**Task Instructions:**
1. Review the existing user profile below
2. Analyze the new conversation for evidence of the 90 dimensions above
3. Update and integrate the findings into a comprehensive user profile
4. For each dimension that can be identified, use the format: Dimension ( Level(High/Medium/Low) )
5. Include brief reasoning for each dimension when possible
6. Maintain existing insights from the old profile while incorporating new observations
7. If a dimension cannot be inferred from either the old profile or new conversation, do not include it

**Existing User Profile:**
{existing_user_profile}

**Latest User-AI Conversation:**
{conversation}

**Updated User Profile:**
Please provide the comprehensive updated user profile below, combining insights from both the existing profile and new conversation:"""

KNOWLEDGE_SYSTEM = """You are a knowledge extraction assistant.
Extract user private data and assistant knowledge from explicit conversation evidence.
1. User private data: personal information, preferences or private facts about the user.
2. Assistant knowledge: explicit statements about what the assistant actually did,
   provided or demonstrated. Do not invent abilities, procedures or general advice.
Be extremely concise and factual. Preserve entities, time and context when present.
Return only JSON {"user_knowledge": ["brief personal fact with context"],
"assistant_knowledge": ["Assistant action at time/context"]}.
Use an empty array for a category with no supporting evidence."""
KNOWLEDGE_USER = """Extract user private data and assistant knowledge from the latest user-AI conversation below.
Keep each personal fact concise. Describe assistant knowledge as an action the
assistant actually provided or demonstrated, with time or context when available.

Latest User-AI Conversation:
{conversation}
"""
