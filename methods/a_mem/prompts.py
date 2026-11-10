"""Prompts/schemas from agiresearch/A-mem ceffb860 (MIT; see UPSTREAM_LICENSE)."""

ANALYSIS_PROMPT = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """

EVOLUTION_PROMPT = """
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.  

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                """

ANALYSIS_FORMAT = {'type': 'json_schema',
 'json_schema': {'name': 'response',
                 'schema': {'type': 'object',
                            'properties': {'keywords': {'type': 'array',
                                                        'items': {'type': 'string'}},
                                           'context': {'type': 'string'},
                                           'tags': {'type': 'array',
                                                    'items': {'type': 'string'}}}}}}

EVOLUTION_FORMAT = {'type': 'json_schema',
 'json_schema': {'name': 'response',
                 'schema': {'type': 'object',
                            'properties': {'should_evolve': {'type': 'boolean'},
                                           'actions': {'type': 'array',
                                                       'items': {'type': 'string'}},
                                           'suggested_connections': {'type': 'array',
                                                                     'items': {'type': 'string'}},
                                           'new_context_neighborhood': {'type': 'array',
                                                                        'items': {'type': 'string'}},
                                           'tags_to_update': {'type': 'array',
                                                              'items': {'type': 'string'}},
                                           'new_tags_neighborhood': {'type': 'array',
                                                                     'items': {'type': 'array',
                                                                               'items': {'type': 'string'}}}},
                            'required': ['should_evolve',
                                         'actions',
                                         'suggested_connections',
                                         'tags_to_update',
                                         'new_context_neighborhood',
                                         'new_tags_neighborhood'],
                            'additionalProperties': False},
                 'strict': True}}
