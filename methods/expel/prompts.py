# Built-in ExpeL prompts adapted from e41ec9a2 (Apache-2.0).
FORMAT_RULES_OPERATION_TEMPLATE = """<OPERATION> <RULE NUMBER>: <RULE>

The available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD <NEW RULE NUMBER>: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. """

human_critique_existing_rules_all_success_template = """{instruction}
Here are the trials:
{success_history}

Here are the EXISTING RULES:
{existing_rules}

By examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:

""" + FORMAT_RULES_OPERATION_TEMPLATE

human_critique_existing_rules_template = """{instruction}
Here are the two previous trials to compare and critique:
TRIAL TASK:
{task}

SUCCESSFUL TRIAL:
{success_history}

FAILED TRIAL:
{fail_history}

Here are the EXISTING RULES:
{existing_rules}

By examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:

""" + FORMAT_RULES_OPERATION_TEMPLATE

INSTRUCTIONS = {'alfworld': {'compare': 'You will be given two previous task trials in which you were placed in a '
                         'household environment and a task to complete: one successful and one '
                         'unsuccessful trial. You failed the trial because you reached the maximum '
                         'allowed number of steps without completing the task.',
              'success': 'You will be given successful tasks trials in which you were placed in a '
                         'household environment and tasks to complete.'},
 'hotpotqa': {'compare': 'You will be given two previous task trials in which you were given '
                         'access to a Docstore API environment and a question to answer: one '
                         'successful and one unsuccessful trial. You failed the trial either '
                         'because you guessed the wrong answer with Finish[<answer>], or you used '
                         'up your set number of reasoning steps.',
              'success': 'You will be given successful tasks trials in which you were given access '
                         'to a Docstore API environment and a question to answer.'},
 'webshop': {'compare': 'You will be given two previous task trials in which you could interact '
                        'with an online store website in order to buy the requested item: one '
                        'successful and one unsuccessful trial. You failed the trial either '
                        'because you reached the maximum allowed number of steps or you bought a '
                        'wrong item.',
             'success': 'You will be given successful task trials in which you could interact with '
                        'an online store website in order to buy the requested items.'}}

INSTRUCTIONS["medicine"] = {
    "compare": "Compare a successful and a failed simulated clinical encounter for the same observable task. "
               "Explain differences in questioning, evidence use, examination and treatment. Induce reusable "
               "conditional clinical rules from observed trajectories and feedback; avoid patient-specific answer memorization.",
    "success": "Analyze successful simulated clinical encounters and induce reusable conditional rules for "
               "questioning, evidence use, examination and treatment. Use the observed trajectories and feedback. "
               "Generalize across patients and do not invent clinical facts.",
}

FEWSHOT_TOKEN_LIMITS = {'hotpotqa': 349, 'alfworld': 1233, 'webshop': 936}

CRITIQUE_SUMMARY_SUFFIX = {
    'full': 'Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:\n',
    'not_full': 'Below are the operations you do to the above list of EXISTING RULES:\n',
}

RULE_TEMPLATES = {
    'hotpotqa': 'The following are some experience you gather on a similar task of question answering using Wikipedia API. Use these as references to help you perform this task:\n{rules}\n',
    'alfworld': 'The following are some experience you gather on a similar task of completing a household task by interacting in a household environment. Use these as references to help you perform this task:\n{rules}\n',
    'webshop': '\nThe following are some experiences (in decreasing order of importance) you gathered on tasks of purchasing items requested by an user by interacting with an online website. Use these experiences as useful references to help you perform better on this task:\n{rules}\n',
    'medicine': 'Use these conditional experiences from simulated clinical tasks as references:\n{rules}\n',
}
