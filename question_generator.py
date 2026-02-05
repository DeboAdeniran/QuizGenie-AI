import spacy
import nltk
from nltk.tokenize import sent_tokenize
import random
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
import time
from loguru import logger
from sentence_transformers import SentenceTransformer
import numpy as np

class ImprovedQuestionGenerator:
    """
    Enhanced intelligent question generator - creates CONCEPTUAL quiz questions
    that test understanding, not just memorization
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize the question generator"""
        try:
            self.nlp = spacy.load(spacy_model)
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Enhanced ImprovedQuestionGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ImprovedQuestionGenerator: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if generator is ready"""
        return self.nlp is not None and self.sentence_model is not None
    
    def generate(self,
                 text: str,
                 num_questions: int = 10,
                 difficulty: str = "mixed",
                 question_types: List[str] = None,
                 topics: Optional[List[str]] = None,
                 context_aware: bool = True) -> Dict[str, Any]:
        """Generate conceptual, understanding-based questions"""
        start_time = time.time()
        
        if question_types is None:
            question_types = ["multipleChoice", "trueFalse"]
        
        # Clean and process text
        text = self._clean_text(text)
        doc = self.nlp(text)
        
        # Build comprehensive knowledge base
        kb = self._build_knowledge_base(doc, text)
        
        logger.info(f"📚 Knowledge Base Built:")
        logger.info(f"  - {len(kb['definitions'])} definitions")
        logger.info(f"  - {len(kb['key_concepts'])} key concepts")
        logger.info(f"  - {len(kb['processes'])} processes")
        logger.info(f"  - {len(kb['relationships'])} relationships")
        logger.info(f"  - {len(kb['historical_facts'])} historical facts")
        logger.info(f"  - {len(kb['roles'])} roles/responsibilities")
        
        all_questions = []
        
        # Generate different question types with CONCEPTUAL focus
        if "multipleChoice" in question_types:
            # Definition questions (reformulated conceptually)
            mcqs_def = self._generate_conceptual_definition_mcqs(kb, num_questions)
            all_questions.extend(mcqs_def)
            
            # Process/Stage questions
            mcqs_process = self._generate_process_mcqs(kb, num_questions)
            all_questions.extend(mcqs_process)
            
            # Role/Responsibility questions
            mcqs_role = self._generate_role_mcqs(kb, num_questions)
            all_questions.extend(mcqs_role)
            
            # Historical/Timeline questions
            mcqs_hist = self._generate_historical_mcqs(kb, num_questions)
            all_questions.extend(mcqs_hist)
            
            # Relationship questions
            mcqs_rel = self._generate_relationship_mcqs(kb, num_questions)
            all_questions.extend(mcqs_rel)
            
            # "Which is NOT" questions
            mcqs_not = self._generate_negative_mcqs(kb, num_questions)
            all_questions.extend(mcqs_not)
            
            # NEW: Comparison questions
            mcqs_comp = self._generate_comparison_mcqs(kb, num_questions)
            all_questions.extend(mcqs_comp)
            
            # NEW: Cause-Effect questions
            mcqs_cause = self._generate_cause_effect_mcqs(kb, num_questions)
            all_questions.extend(mcqs_cause)
            
            # NEW: Inference questions
            mcqs_infer = self._generate_inference_mcqs(kb, num_questions)
            all_questions.extend(mcqs_infer)
            
            # NEW: Application questions
            mcqs_app = self._generate_application_mcqs(kb, num_questions)
            all_questions.extend(mcqs_app)
            
            # NEW: Sequence/Order questions
            mcqs_seq = self._generate_sequence_mcqs(kb, num_questions)
            all_questions.extend(mcqs_seq)
        
        if "trueFalse" in question_types:
            # Conceptual true/false (not just text copying)
            tfs = self._generate_conceptual_true_false(kb, num_questions)
            all_questions.extend(tfs)
        
        if "shortAnswer" in question_types:
            sas = self._generate_short_answer(kb, num_questions)
            all_questions.extend(sas)
        
        # STRICT filtering - only keep GOOD questions
        valid_questions = []
        for q in all_questions:
            if self._is_high_quality_question(q):
                valid_questions.append(q)
        
        logger.info(f"✅ Generated {len(valid_questions)} high-quality questions from {len(all_questions)} attempts")
        
        # Select diverse questions
        final_questions = self._select_diverse_questions(valid_questions, num_questions)
        
        # Apply difficulty
        final_questions = self._set_difficulty(final_questions, difficulty)
        
        generation_time = time.time() - start_time
        
        return {
            "questions": final_questions[:num_questions],
            "generation_time": round(generation_time, 2),
            "total_questions": len(final_questions),
            "difficulty_distribution": {
                "easy": len([q for q in final_questions if q['difficulty'] == 'easy']),
                "medium": len([q for q in final_questions if q['difficulty'] == 'medium']),
                "hard": len([q for q in final_questions if q['difficulty'] == 'hard'])
            },
            "topics_covered": list(set([q.get('topic', 'General') for q in final_questions])),
            "quality_score": np.mean([q['confidence'] for q in final_questions]) if final_questions else 0
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text and remove problematic content"""
        # Remove excessive LaTeX/math notation
        text = re.sub(r'\$[^\$]+\$', '', text)
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Remove figure/table references
        text = re.sub(r'Figure \d+\.?\d*', '', text)
        text = re.sub(r'Table \d+\.?\d*', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s([.,;:!?])', r'\1', text)
        
        return text.strip()
    
    def _build_knowledge_base(self, doc, text: str) -> Dict:
        """Build comprehensive knowledge base with relationships and concepts"""
        kb = {
            'definitions': [],
            'key_concepts': [],
            'processes': [],
            'stages': [],
            'roles': [],
            'historical_facts': [],
            'relationships': [],
            'comparisons': [],
            'examples': [],
            'clean_sentences': [],
            'entities': defaultdict(list),
            'temporal_info': [],
            'cause_effect': [],
            'advantages': [],
            'disadvantages': [],
            'sequences': [],
            'benefits': [],
            'challenges': [],
            'characteristics': []
        }
        
        # Extract clean sentences
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            if self._is_problematic_sentence(sent_text):
                continue
            
            word_count = len(sent_text.split())
            if not (7 <= word_count <= 40):
                continue
            
            if not sent_text[0].isupper() or not sent_text[-1] in '.!?':
                continue
            
            kb['clean_sentences'].append({
                'text': sent_text,
                'doc': sent,
                'word_count': word_count
            })
        
        # Extract definitions (multiple patterns)
        self._extract_definitions(kb)
        
        # Extract processes and stages
        self._extract_processes_and_stages(kb)
        
        # Extract roles and responsibilities
        self._extract_roles(kb)
        
        # Extract historical/temporal information
        self._extract_historical_facts(kb)
        
        # Extract relationships
        self._extract_relationships(kb)
        
        # NEW: Extract comparisons
        self._extract_comparisons(kb)
        
        # NEW: Extract cause-effect relationships
        self._extract_cause_effect(kb)
        
        # NEW: Extract advantages/disadvantages
        self._extract_advantages_disadvantages(kb)
        
        # NEW: Extract examples
        self._extract_examples(kb)
        
        # NEW: Extract sequences/steps
        self._extract_sequences(kb)
        
        # Extract entities by type
        for sent_info in kb['clean_sentences']:
            for ent in sent_info['doc'].ents:
                kb['entities'][ent.label_].append({
                    'text': ent.text,
                    'sentence': sent_info['text']
                })
        
        # Extract key concepts
        concept_freq = Counter()
        for sent_info in kb['clean_sentences']:
            for chunk in sent_info['doc'].noun_chunks:
                if 1 <= len(chunk) <= 4 and not chunk.root.is_stop:
                    concept_freq[chunk.text.lower()] += 1
        
        for concept, freq in concept_freq.most_common(30):
            if freq >= 2:
                kb['key_concepts'].append(concept)
        
        return kb
    
    def _extract_definitions(self, kb: Dict):
        """Extract definitions using multiple patterns"""
        for sent_info in kb['clean_sentences']:
            sent_text = sent_info['text']
            
            # Pattern 1: "X is defined as Y"
            match = re.search(r'^([A-Z][^,]{2,50}?)\s+is\s+defined\s+as\s+([^.]+)\.', sent_text)
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                if self._is_good_definition(term, definition):
                    kb['definitions'].append({
                        'term': term,
                        'definition': definition,
                        'sentence': sent_text,
                        'pattern': 'defined_as'
                    })
                continue
            
            # Pattern 2: "X is/are [a/an/the] Y" (substantive)
            match = re.search(r'^([A-Z][^,]{2,50}?)\s+(is|are)\s+(a|an|the)\s+([^,.]{10,100})(?:[,.])', sent_text)
            if match:
                term = match.group(1).strip()
                definition = match.group(4).strip()
                if self._is_good_definition(term, definition):
                    kb['definitions'].append({
                        'term': term,
                        'definition': definition,
                        'sentence': sent_text,
                        'pattern': 'is_a'
                    })
                continue
            
            # Pattern 3: "X refers to Y"
            match = re.search(r'^([A-Z][^,]{2,50}?)\s+refers?\s+to\s+([^.]+)\.', sent_text)
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                if self._is_good_definition(term, definition):
                    kb['definitions'].append({
                        'term': term,
                        'definition': definition,
                        'sentence': sent_text,
                        'pattern': 'refers_to'
                    })
    
    def _extract_processes_and_stages(self, kb: Dict):
        """Extract processes, stages, and steps"""
        process_keywords = ['process', 'stage', 'step', 'phase', 'lifecycle', 'workflow']
        
        for sent_info in kb['clean_sentences']:
            sent_text = sent_info['text']
            sent_lower = sent_text.lower()
            
            # Check for process/stage mentions
            if any(kw in sent_lower for kw in process_keywords):
                kb['processes'].append({
                    'text': sent_text,
                    'type': 'process_description'
                })
            
            # Pattern: "X stage focuses on/involves Y"
            match = re.search(r'(\w+)\s+stage\s+(focuses on|involves|includes)\s+([^.]+)', sent_text, re.I)
            if match:
                stage_name = match.group(1)
                focus = match.group(3).strip()
                kb['stages'].append({
                    'name': stage_name,
                    'focus': focus,
                    'sentence': sent_text
                })
    
    def _extract_roles(self, kb: Dict):
        """Extract roles and responsibilities"""
        role_patterns = [
            r'([A-Z][a-zA-Z\s]+(?:Engineer|Analyst|Scientist|Manager|Developer))\s+(?:is|are)\s+responsible for\s+([^.]+)',
            r'([A-Z][a-zA-Z\s]+(?:Engineer|Analyst|Scientist|Manager|Developer))\s+(?:builds?|creates?|manages?|analyzes?)\s+([^.]+)',
            r'The role of\s+(?:a|an|the)\s+([^,]+?)\s+is to\s+([^.]+)'
        ]
        
        for sent_info in kb['clean_sentences']:
            sent_text = sent_info['text']
            
            for pattern in role_patterns:
                match = re.search(pattern, sent_text)
                if match:
                    role = match.group(1).strip()
                    responsibility = match.group(2).strip()
                    kb['roles'].append({
                        'role': role,
                        'responsibility': responsibility,
                        'sentence': sent_text
                    })
                    break
    
    def _extract_historical_facts(self, kb: Dict):
        """Extract historical facts and temporal information"""
        # Look for dates and temporal markers
        date_pattern = r'\b(19\d{2}|20\d{2})\b'
        temporal_markers = ['first', 'began', 'started', 'emerged', 'founded', 'invented', 'discovered']
        
        for sent_info in kb['clean_sentences']:
            sent_text = sent_info['text']
            sent_lower = sent_text.lower()
            
            # Has date or temporal marker
            has_date = re.search(date_pattern, sent_text)
            has_temporal = any(marker in sent_lower for marker in temporal_markers)
            
            if has_date or has_temporal:
                # Extract person if mentioned
                persons = [ent.text for ent in sent_info['doc'].ents if ent.label_ == 'PERSON']
                dates = re.findall(date_pattern, sent_text)
                
                kb['historical_facts'].append({
                    'text': sent_text,
                    'persons': persons,
                    'dates': dates,
                    'type': 'historical'
                })
                
                kb['temporal_info'].append({
                    'sentence': sent_text,
                    'dates': dates,
                    'persons': persons
                })
    
    def _extract_relationships(self, kb: Dict):
        """Extract relationships between concepts"""
        relationship_verbs = ['causes', 'leads to', 'results in', 'affects', 'influences', 'enables', 
                             'prevents', 'requires', 'depends on', 'includes', 'consists of']
        
        for sent_info in kb['clean_sentences']:
            sent_text = sent_info['text']
            sent_lower = sent_text.lower()
            
            for verb in relationship_verbs:
                if verb in sent_lower:
                    kb['relationships'].append({
                        'text': sent_text,
                        'relationship_type': verb,
                        'sentence': sent_text
                    })
                    break
    
    def _extract_comparisons(self, kb: Dict):
        """Extract comparison statements"""
        comparison_markers = [
            'more than', 'less than', 'compared to', 'versus', 'vs',
            'while', 'whereas', 'unlike', 'different from', 'similar to',
            'better than', 'worse than', 'faster than', 'slower than'
        ]
        
        for sent_info in kb['clean_sentences']:
            sent_text = sent_info['text']
            sent_lower = sent_text.lower()
            
            for marker in comparison_markers:
                if marker in sent_lower:
                    # Extract the two things being compared
                    doc = sent_info['doc']
                    entities = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4]
                    
                    if len(entities) >= 2:
                        kb['comparisons'].append({
                            'text': sent_text,
                            'marker': marker,
                            'entities': entities[:2],
                            'sentence': sent_text
                        })
                    break
    
    def _extract_cause_effect(self, kb: Dict):
        """Extract cause-effect relationships"""
        cause_markers = [
            'because', 'due to', 'as a result of', 'caused by',
            'leads to', 'results in', 'causes', 'therefore',
            'consequently', 'thus', 'hence'
        ]
        
        for sent_info in kb['clean_sentences']:
            sent_text = sent_info['text']
            sent_lower = sent_text.lower()
            
            for marker in cause_markers:
                if marker in sent_lower:
                    # Try to split into cause and effect
                    parts = re.split(marker, sent_text, maxsplit=1, flags=re.IGNORECASE)
                    
                    if len(parts) == 2:
                        if marker in ['because', 'due to', 'as a result of', 'caused by']:
                            # Effect comes first, cause second
                            effect = parts[0].strip()
                            cause = parts[1].strip()
                        else:
                            # Cause comes first, effect second
                            cause = parts[0].strip()
                            effect = parts[1].strip()
                        
                        kb['cause_effect'].append({
                            'cause': cause,
                            'effect': effect,
                            'marker': marker,
                            'sentence': sent_text
                        })
                    break
    
    def _extract_advantages_disadvantages(self, kb: Dict):
        """Extract advantages, disadvantages, benefits, and challenges"""
        advantage_markers = [
            'advantage', 'benefit', 'strength', 'pro', 'positive',
            'improves', 'enhances', 'enables', 'facilitates'
        ]
        
        disadvantage_markers = [
            'disadvantage', 'drawback', 'limitation', 'weakness', 'con',
            'challenge', 'problem', 'issue', 'difficulty', 'threat'
        ]
        
        for sent_info in kb['clean_sentences']:
            sent_text = sent_info['text']
            sent_lower = sent_text.lower()
            
            # Check for advantages/benefits
            if any(marker in sent_lower for marker in advantage_markers):
                kb['advantages'].append({
                    'text': sent_text,
                    'type': 'advantage'
                })
                kb['benefits'].append(sent_text)
            
            # Check for disadvantages/challenges
            if any(marker in sent_lower for marker in disadvantage_markers):
                kb['disadvantages'].append({
                    'text': sent_text,
                    'type': 'disadvantage'
                })
                kb['challenges'].append(sent_text)
    
    def _extract_examples(self, kb: Dict):
        """Extract examples and specific instances"""
        example_markers = [
            'for example', 'such as', 'for instance', 'including',
            'e.g.', 'like', 'particularly', 'especially'
        ]
        
        for sent_info in kb['clean_sentences']:
            sent_text = sent_info['text']
            sent_lower = sent_text.lower()
            
            for marker in example_markers:
                if marker in sent_lower:
                    # Try to extract what is being exemplified and the examples
                    parts = re.split(marker, sent_text, maxsplit=1, flags=re.IGNORECASE)
                    
                    if len(parts) == 2:
                        concept = parts[0].strip()
                        examples_text = parts[1].strip()
                        
                        # Extract individual examples
                        examples_list = re.split(r',\s*(?:and\s+)?', examples_text.rstrip('.'))
                        
                        kb['examples'].append({
                            'concept': concept,
                            'examples': [ex.strip() for ex in examples_list if ex.strip()],
                            'sentence': sent_text,
                            'marker': marker
                        })
                    break
    
    def _extract_sequences(self, kb: Dict):
        """Extract sequences, steps, or ordered processes"""
        sequence_markers = [
            'first', 'second', 'third', 'finally', 'then', 'next',
            'step 1', 'step 2', 'stage 1', 'stage 2',
            'initially', 'subsequently', 'lastly'
        ]
        
        sequence_sentences = []
        
        for sent_info in kb['clean_sentences']:
            sent_text = sent_info['text']
            sent_lower = sent_text.lower()
            
            # Check if sentence contains sequence markers
            for marker in sequence_markers:
                if marker in sent_lower:
                    sequence_sentences.append({
                        'text': sent_text,
                        'marker': marker,
                        'position': self._extract_sequence_position(marker)
                    })
                    break
        
        # Sort by position if possible
        if sequence_sentences:
            sequence_sentences.sort(key=lambda x: x['position'])
            kb['sequences'] = sequence_sentences
    
    def _extract_sequence_position(self, marker: str) -> int:
        """Extract numeric position from sequence marker"""
        marker_lower = marker.lower()
        
        position_map = {
            'first': 1, 'initially': 1, 'step 1': 1, 'stage 1': 1,
            'second': 2, 'step 2': 2, 'stage 2': 2,
            'third': 3, 'step 3': 3, 'stage 3': 3,
            'fourth': 4, 'step 4': 4, 'stage 4': 4,
            'then': 5, 'next': 5, 'subsequently': 6,
            'finally': 99, 'lastly': 99
        }
        
        return position_map.get(marker_lower, 50)
    
    def _is_problematic_sentence(self, sent: str) -> bool:
        """Check if sentence is problematic for questions"""
        if sent.count('=') > 2 or sent.count('(') > 3:
            return True
        if '\\' in sent or '{' in sent or '}' in sent:
            return True
        symbol_ratio = sum(1 for c in sent if not c.isalnum() and c not in ' .,!?-\'\"') / max(len(sent), 1)
        if symbol_ratio > 0.15:
            return True
        if sent.startswith(('And ', 'But ', 'Or ', 'So ', 'However, ')):
            return True
        if re.search(r'\[\d+\]', sent) or re.search(r'\(\d{4}\)', sent):
            return True
        return False
    
    def _is_good_definition(self, term: str, definition: str) -> bool:
        """Check if a definition is usable"""
        if not (2 <= len(term.split()) <= 10):
            return False
        if not (5 <= len(definition.split()) <= 40):
            return False
        if definition.count('(') > 2 or definition.count('=') > 1:
            return False
        return True
    
    # ========== CONCEPTUAL MCQ GENERATION ==========
    
    def _generate_conceptual_definition_mcqs(self, kb: Dict, target: int) -> List[Dict]:
        """Generate 'Which best describes' style definition questions"""
        questions = []
        
        for def_info in kb['definitions'][:target]:
            term = def_info['term']
            correct_def = def_info['definition']
            
            # Reformulate question conceptually
            question_stems = [
                f"Which of the following best describes {term}?",
                f"What is {term}?",
                f"{term} can best be defined as:",
                f"Which statement accurately describes {term}?"
            ]
            
            question_text = random.choice(question_stems)
            
            # Get plausible distractors
            distractors = self._generate_smart_distractors(correct_def, kb, 3)
            
            if len(distractors) < 3:
                continue
            
            # Format options as complete sentences
            correct_answer = self._format_as_option(correct_def)
            distractor_options = [self._format_as_option(d) for d in distractors]
            
            options = [correct_answer] + distractor_options
            random.shuffle(options)
            
            questions.append({
                "type": "multipleChoice",
                "text": question_text,
                "options": options,
                "correct_answer": correct_answer,
                "explanation": f"According to the text: \"{def_info['sentence']}\"",
                "difficulty": "medium",
                "points": 5,
                "topic": "Concepts & Definitions",
                "source_context": def_info['sentence'],
                "confidence": 0.88,
                "keywords": [term]
            })
        
        return questions
    
    def _generate_process_mcqs(self, kb: Dict, target: int) -> List[Dict]:
        """Generate questions about stages/processes"""
        questions = []
        
        for stage_info in kb['stages'][:target]:
            stage_name = stage_info['name']
            focus = stage_info['focus']
            
            question_text = f"Which of the following best describes the focus of the {stage_name} stage?"
            
            # Get distractors from other stages
            distractors = []
            for other_stage in kb['stages']:
                if other_stage != stage_info:
                    distractors.append(other_stage['focus'])
            
            # Add generic distractors if needed
            generic_distractors = [
                "data visualization and reporting",
                "model deployment and monitoring",
                "requirement gathering and planning",
                "security and compliance management"
            ]
            
            for gd in generic_distractors:
                if gd not in distractors and len(distractors) < 3:
                    distractors.append(gd)
            
            if len(distractors) < 3:
                continue
            
            correct_answer = self._format_as_option(focus)
            distractor_options = [self._format_as_option(d) for d in distractors[:3]]
            
            options = [correct_answer] + distractor_options
            random.shuffle(options)
            
            questions.append({
                "type": "multipleChoice",
                "text": question_text,
                "options": options,
                "correct_answer": correct_answer,
                "explanation": f"The text indicates: \"{stage_info['sentence']}\"",
                "difficulty": "medium",
                "points": 5,
                "topic": "Processes & Stages",
                "source_context": stage_info['sentence'],
                "confidence": 0.85,
                "keywords": [stage_name]
            })
        
        return questions
    
    def _generate_role_mcqs(self, kb: Dict, target: int) -> List[Dict]:
        """Generate questions about roles and responsibilities"""
        questions = []
        
        for role_info in kb['roles'][:target]:
            role = role_info['role']
            responsibility = role_info['responsibility']
            
            question_text = f"Which role is primarily responsible for {responsibility}?"
            
            # Get other roles as distractors
            distractors = []
            for other_role in kb['roles']:
                if other_role != role_info:
                    distractors.append(other_role['role'])
            
            # Add common role distractors
            common_roles = ["Data Analyst", "Data Scientist", "Data Engineer", 
                          "Business Analyst", "ML Engineer", "Database Administrator"]
            
            for cr in common_roles:
                if cr not in distractors and cr != role and len(distractors) < 3:
                    distractors.append(cr)
            
            if len(distractors) < 3:
                continue
            
            options = [role] + distractors[:3]
            random.shuffle(options)
            
            questions.append({
                "type": "multipleChoice",
                "text": question_text,
                "options": options,
                "correct_answer": role,
                "explanation": f"According to the text: \"{role_info['sentence']}\"",
                "difficulty": "medium",
                "points": 5,
                "topic": "Roles & Responsibilities",
                "source_context": role_info['sentence'],
                "confidence": 0.87,
                "keywords": [role]
            })
        
        return questions
    
    def _generate_historical_mcqs(self, kb: Dict, target: int) -> List[Dict]:
        """Generate historical/timeline questions"""
        questions = []
        
        for hist_info in kb['historical_facts'][:target]:
            if not hist_info['persons'] and not hist_info['dates']:
                continue
            
            # Question type 1: Who did X?
            if hist_info['persons']:
                person = hist_info['persons'][0]
                
                # Extract what they did
                sent = hist_info['text']
                
                question_text = f"Who {self._extract_action_from_sentence(sent, person)}?"
                
                # Get other persons as distractors
                all_persons = set()
                for h in kb['historical_facts']:
                    all_persons.update(h['persons'])
                
                distractors = [p for p in all_persons if p != person]
                
                # Add famous names if needed
                famous_names = ["Alan Turing", "John McCarthy", "Geoffrey Hinton", 
                               "Yann LeCun", "Andrew Ng"]
                for fn in famous_names:
                    if fn not in distractors and fn != person and len(distractors) < 3:
                        distractors.append(fn)
                
                if len(distractors) >= 3:
                    options = [person] + distractors[:3]
                    random.shuffle(options)
                    
                    questions.append({
                        "type": "multipleChoice",
                        "text": question_text,
                        "options": options,
                        "correct_answer": person,
                        "explanation": f"The text states: \"{hist_info['text']}\"",
                        "difficulty": "hard",
                        "points": 7,
                        "topic": "History & Timeline",
                        "source_context": hist_info['text'],
                        "confidence": 0.82,
                        "keywords": [person]
                    })
            
            # Question type 2: When did X happen?
            if hist_info['dates']:
                date = hist_info['dates'][0]
                event = self._extract_event_from_sentence(hist_info['text'], date)
                
                if event:
                    question_text = f"When {event}?"
                    
                    # Generate plausible year distractors
                    try:
                        year = int(date)
                        distractors = [
                            str(year - random.randint(5, 15)),
                            str(year + random.randint(5, 15)),
                            str(year - random.randint(20, 30))
                        ]
                        
                        options = [date] + distractors
                        random.shuffle(options)
                        
                        questions.append({
                            "type": "multipleChoice",
                            "text": question_text,
                            "options": options,
                            "correct_answer": date,
                            "explanation": f"The text mentions: \"{hist_info['text']}\"",
                            "difficulty": "hard",
                            "points": 7,
                            "topic": "History & Timeline",
                            "source_context": hist_info['text'],
                            "confidence": 0.80,
                            "keywords": [date]
                        })
                    except:
                        pass
        
        return questions
    
    def _generate_relationship_mcqs(self, kb: Dict, target: int) -> List[Dict]:
        """Generate questions about relationships between concepts"""
        questions = []
        
        for rel_info in kb['relationships'][:target]:
            sent = rel_info['text']
            rel_type = rel_info['relationship_type']
            
            # Extract subject and object
            doc = self.nlp(sent)
            
            # Find main entities/concepts
            entities = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
            
            if len(entities) >= 2:
                subject = entities[0]
                obj = entities[1]
                
                question_text = f"According to the text, what does {subject} {rel_type}?"
                
                # Get other objects as distractors
                distractors = []
                for other_rel in kb['relationships']:
                    other_doc = self.nlp(other_rel['text'])
                    other_entities = [chunk.text for chunk in other_doc.noun_chunks if len(chunk.text.split()) <= 3]
                    for ent in other_entities:
                        if ent != obj and ent not in distractors:
                            distractors.append(ent)
                            if len(distractors) >= 3:
                                break
                    if len(distractors) >= 3:
                        break
                
                if len(distractors) >= 3:
                    options = [obj] + distractors[:3]
                    random.shuffle(options)
                    
                    questions.append({
                        "type": "multipleChoice",
                        "text": question_text,
                        "options": options,
                        "correct_answer": obj,
                        "explanation": f"The text states: \"{sent}\"",
                        "difficulty": "medium",
                        "points": 5,
                        "topic": "Relationships & Connections",
                        "source_context": sent,
                        "confidence": 0.75,
                        "keywords": [subject, obj]
                    })
        
        return questions
    
    def _generate_negative_mcqs(self, kb: Dict, target: int) -> List[Dict]:
        """Generate 'Which is NOT' style questions"""
        questions = []
        
        # Group related concepts
        concept_groups = self._group_related_concepts(kb)
        
        for group_name, concepts in list(concept_groups.items())[:target]:
            if len(concepts) < 4:
                continue
            
            # Pick 3 true items and 1 false item
            true_items = random.sample(concepts, min(3, len(concepts)))
            
            # Find a plausible but false item
            false_item = self._find_false_item(true_items, kb)
            
            if not false_item:
                continue
            
            question_text = f"Which of the following is NOT mentioned as part of {group_name}?"
            
            options = true_items + [false_item]
            random.shuffle(options)
            
            questions.append({
                "type": "multipleChoice",
                "text": question_text,
                "options": options,
                "correct_answer": false_item,
                "explanation": f"{false_item} is not mentioned in relation to {group_name} in the text.",
                "difficulty": "medium",
                "points": 5,
                "topic": "Comprehension",
                "source_context": f"Related to {group_name}",
                "confidence": 0.78,
                "keywords": [group_name]
            })
        
        return questions
    
    def _generate_conceptual_true_false(self, kb: Dict, target: int) -> List[Dict]:
        """Generate conceptual T/F questions (not just copying text)"""
        questions = []
        
        # Generate TRUE questions (paraphrased)
        for sent_info in kb['clean_sentences'][:target]:
            if len(questions) >= target:
                break
            
            original = sent_info['text']
            
            # Paraphrase for TRUE questions
            paraphrased = self._paraphrase_sentence(original, sent_info['doc'])
            
            if paraphrased and paraphrased != original:
                questions.append({
                    "type": "trueFalse",
                    "text": paraphrased,
                    "options": ["True", "False"],
                    "correct_answer": "True",
                    "explanation": f"This is correct based on: \"{original}\"",
                    "difficulty": "easy",
                    "points": 2,
                    "topic": "Comprehension",
                    "source_context": original,
                    "confidence": 0.85,
                    "keywords": []
                })
        
        # Generate FALSE questions (intelligent modifications)
        for sent_info in kb['clean_sentences'][:target]:
            if len(questions) >= target * 2:
                break
            
            original = sent_info['text']
            false_version = self._create_intelligent_false_statement(original, sent_info['doc'], kb)
            
            if false_version and false_version != original:
                questions.append({
                    "type": "trueFalse",
                    "text": false_version,
                    "options": ["True", "False"],
                    "correct_answer": "False",
                    "explanation": f"This is incorrect. The text states: \"{original}\"",
                    "difficulty": "medium",
                    "points": 3,
                    "topic": "Comprehension",
                    "source_context": original,
                    "confidence": 0.82,
                    "keywords": []
                })
        
        return questions
    
    def _generate_short_answer(self, kb: Dict, target: int) -> List[Dict]:
        """Generate short answer questions"""
        questions = []
        
        for def_info in kb['definitions'][:target]:
            questions.append({
                "type": "shortAnswer",
                "text": f"What is {def_info['term']}?",
                "options": None,
                "correct_answer": def_info['definition'],
                "explanation": f"From the text: \"{def_info['sentence']}\"",
                "difficulty": "medium",
                "points": 5,
                "topic": "Definitions",
                "source_context": def_info['sentence'],
                "confidence": 0.88,
                "keywords": def_info['term'].split()
            })
        
        return questions
    
    # ========== NEW ADVANCED QUESTION TYPES ==========
    
    def _generate_comparison_mcqs(self, kb: Dict, target: int) -> List[Dict]:
        """Generate comparison questions"""
        questions = []
        
        for comp_info in kb['comparisons'][:target]:
            entities = comp_info['entities']
            marker = comp_info['marker']
            
            if len(entities) >= 2:
                entity1 = entities[0]
                entity2 = entities[1]
                
                # Generate comparison question
                question_stems = [
                    f"How does {entity1} compare to {entity2} according to the text?",
                    f"What is the difference between {entity1} and {entity2}?",
                    f"According to the text, {entity1} is different from {entity2} in that:"
                ]
                
                question_text = random.choice(question_stems)
                
                # The correct answer is the comparison statement
                correct_answer = comp_info['text']
                
                # Generate plausible distractors
                distractors = []
                
                # Get other comparison statements as distractors
                for other_comp in kb['comparisons']:
                    if other_comp != comp_info:
                        distractors.append(other_comp['text'])
                
                # Add generic distractors if needed
                if len(distractors) < 3:
                    generic = [
                        f"{entity1} and {entity2} are identical in function",
                        f"{entity1} is always preferred over {entity2}",
                        f"There is no significant difference between them"
                    ]
                    distractors.extend(generic)
                
                if len(distractors) >= 3:
                    options = [correct_answer] + distractors[:3]
                    random.shuffle(options)
                    
                    questions.append({
                        "type": "multipleChoice",
                        "text": question_text,
                        "options": options,
                        "correct_answer": correct_answer,
                        "explanation": f"The text states: \"{comp_info['sentence']}\"",
                        "difficulty": "medium",
                        "points": 5,
                        "topic": "Comparison & Contrast",
                        "source_context": comp_info['sentence'],
                        "confidence": 0.83,
                        "keywords": entities
                    })
        
        return questions
    
    def _generate_cause_effect_mcqs(self, kb: Dict, target: int) -> List[Dict]:
        """Generate cause-effect questions"""
        questions = []
        
        for ce_info in kb['cause_effect'][:target]:
            cause = ce_info['cause']
            effect = ce_info['effect']
            
            # Question type 1: What causes X?
            if len(cause.split()) <= 20:
                question_text = f"According to the text, what causes {effect.rstrip('.?!')}?"
                correct_answer = cause.strip('.?!')
                
                # Get other causes as distractors
                distractors = []
                for other_ce in kb['cause_effect']:
                    if other_ce != ce_info and len(other_ce['cause'].split()) <= 20:
                        distractors.append(other_ce['cause'].strip('.?!'))
                
                if len(distractors) >= 3:
                    options = [correct_answer] + distractors[:3]
                    random.shuffle(options)
                    
                    questions.append({
                        "type": "multipleChoice",
                        "text": question_text,
                        "options": options,
                        "correct_answer": correct_answer,
                        "explanation": f"The text indicates: \"{ce_info['sentence']}\"",
                        "difficulty": "medium",
                        "points": 5,
                        "topic": "Cause & Effect",
                        "source_context": ce_info['sentence'],
                        "confidence": 0.84,
                        "keywords": []
                    })
            
            # Question type 2: What is the effect of X?
            if len(effect.split()) <= 20:
                question_text = f"What is the result of {cause.rstrip('.?!')}?"
                correct_answer = effect.strip('.?!')
                
                # Get other effects as distractors
                distractors = []
                for other_ce in kb['cause_effect']:
                    if other_ce != ce_info and len(other_ce['effect'].split()) <= 20:
                        distractors.append(other_ce['effect'].strip('.?!'))
                
                if len(distractors) >= 3:
                    options = [correct_answer] + distractors[:3]
                    random.shuffle(options)
                    
                    questions.append({
                        "type": "multipleChoice",
                        "text": question_text,
                        "options": options,
                        "correct_answer": correct_answer,
                        "explanation": f"The text states: \"{ce_info['sentence']}\"",
                        "difficulty": "medium",
                        "points": 5,
                        "topic": "Cause & Effect",
                        "source_context": ce_info['sentence'],
                        "confidence": 0.84,
                        "keywords": []
                    })
        
        return questions
    
    def _generate_inference_mcqs(self, kb: Dict, target: int) -> List[Dict]:
        """Generate inference questions that require understanding beyond literal text"""
        questions = []
        
        # Inference from advantages
        for adv_info in kb['advantages'][:target]:
            text = adv_info['text']
            
            question_stems = [
                "Based on the text, which of the following can be inferred?",
                "What can be concluded from the information provided?",
                "The text suggests that:"
            ]
            
            question_text = random.choice(question_stems)
            
            # Create inference-based answer
            correct_answer = self._create_inference(text)
            
            if correct_answer:
                # Get plausible distractors
                distractors = self._generate_inference_distractors(correct_answer, kb)
                
                if len(distractors) >= 3:
                    options = [correct_answer] + distractors[:3]
                    random.shuffle(options)
                    
                    questions.append({
                        "type": "multipleChoice",
                        "text": question_text,
                        "options": options,
                        "correct_answer": correct_answer,
                        "explanation": f"This can be inferred from: \"{text}\"",
                        "difficulty": "hard",
                        "points": 7,
                        "topic": "Inference & Analysis",
                        "source_context": text,
                        "confidence": 0.75,
                        "keywords": []
                    })
        
        # Inference from relationships
        for rel_info in kb['relationships'][:target]:
            if len(questions) >= target:
                break
            
            text = rel_info['text']
            rel_type = rel_info['relationship_type']
            
            # Create "why" or "implication" question
            question_text = f"Based on the text, what can be inferred about the relationship described?"
            
            correct_answer = f"It demonstrates how different concepts {rel_type} one another"
            
            distractors = [
                "There is no clear relationship between the concepts",
                "The concepts are completely independent",
                "The relationship is purely coincidental"
            ]
            
            options = [correct_answer] + distractors
            random.shuffle(options)
            
            questions.append({
                "type": "multipleChoice",
                "text": question_text,
                "options": options,
                "correct_answer": correct_answer,
                "explanation": f"This is implied by: \"{text}\"",
                "difficulty": "hard",
                "points": 7,
                "topic": "Inference & Analysis",
                "source_context": text,
                "confidence": 0.72,
                "keywords": []
            })
        
        return questions
    
    def _generate_application_mcqs(self, kb: Dict, target: int) -> List[Dict]:
        """Generate application questions - applying concepts to scenarios"""
        questions = []
        
        # Application of definitions
        for def_info in kb['definitions'][:target]:
            term = def_info['term']
            definition = def_info['definition']
            
            question_text = f"In which scenario would you most likely use or encounter {term}?"
            
            # Create scenario-based correct answer
            correct_scenario = self._create_application_scenario(term, definition)
            
            if correct_scenario:
                # Create distractor scenarios
                distractors = self._create_distractor_scenarios(term, kb)
                
                if len(distractors) >= 3:
                    options = [correct_scenario] + distractors[:3]
                    random.shuffle(options)
                    
                    questions.append({
                        "type": "multipleChoice",
                        "text": question_text,
                        "options": options,
                        "correct_answer": correct_scenario,
                        "explanation": f"Based on the definition: \"{definition}\"",
                        "difficulty": "hard",
                        "points": 7,
                        "topic": "Application & Practice",
                        "source_context": def_info['sentence'],
                        "confidence": 0.77,
                        "keywords": [term]
                    })
        
        # Application of processes
        for process_info in kb['processes'][:target]:
            if len(questions) >= target:
                break
            
            text = process_info['text']
            
            question_text = "When would this process be most applicable?"
            
            correct_answer = self._extract_process_context(text)
            
            if correct_answer:
                distractors = [
                    "When performing basic data entry tasks",
                    "During system maintenance and updates",
                    "When creating documentation only"
                ]
                
                options = [correct_answer] + distractors
                random.shuffle(options)
                
                questions.append({
                    "type": "multipleChoice",
                    "text": question_text,
                    "options": options,
                    "correct_answer": correct_answer,
                    "explanation": f"Based on: \"{text}\"",
                    "difficulty": "hard",
                    "points": 7,
                    "topic": "Application & Practice",
                    "source_context": text,
                    "confidence": 0.74,
                    "keywords": []
                })
        
        return questions
    
    def _generate_sequence_mcqs(self, kb: Dict, target: int) -> List[Dict]:
        """Generate questions about order and sequence"""
        questions = []
        
        if len(kb['sequences']) >= 3:
            # Create "what comes first" type questions
            sequences = kb['sequences'][:target]
            
            if len(sequences) >= 3:
                # Sort by position
                sorted_seqs = sorted(sequences, key=lambda x: x['position'])
                
                question_text = "What is the correct order of steps/stages mentioned in the text?"
                
                # Create correct sequence
                correct_order = [seq['text'] for seq in sorted_seqs[:4]]
                correct_answer = " → ".join([self._shorten_text(s, 40) for s in correct_order])
                
                # Create scrambled versions as distractors
                distractors = []
                for _ in range(3):
                    scrambled = correct_order.copy()
                    random.shuffle(scrambled)
                    if scrambled != correct_order:
                        dist = " → ".join([self._shorten_text(s, 40) for s in scrambled])
                        if dist not in distractors:
                            distractors.append(dist)
                
                if len(distractors) >= 3:
                    options = [correct_answer] + distractors[:3]
                    random.shuffle(options)
                    
                    questions.append({
                        "type": "multipleChoice",
                        "text": question_text,
                        "options": options,
                        "correct_answer": correct_answer,
                        "explanation": "This is the order presented in the text.",
                        "difficulty": "hard",
                        "points": 7,
                        "topic": "Sequence & Order",
                        "source_context": " ".join(correct_order),
                        "confidence": 0.80,
                        "keywords": []
                    })
        
        # Create "what comes next" questions
        for i in range(len(kb['sequences']) - 1):
            if len(questions) >= target:
                break
            
            current = kb['sequences'][i]
            next_item = kb['sequences'][i + 1]
            
            question_text = f"After {self._shorten_text(current['text'], 50)}, what comes next?"
            
            correct_answer = self._shorten_text(next_item['text'], 60)
            
            # Get other sequence items as distractors
            distractors = []
            for seq in kb['sequences']:
                if seq != next_item and seq != current:
                    dist = self._shorten_text(seq['text'], 60)
                    distractors.append(dist)
                    if len(distractors) >= 3:
                        break
            
            if len(distractors) >= 3:
                options = [correct_answer] + distractors[:3]
                random.shuffle(options)
                
                questions.append({
                    "type": "multipleChoice",
                    "text": question_text,
                    "options": options,
                    "correct_answer": correct_answer,
                    "explanation": f"The text indicates this follows: \"{current['text']}\"",
                    "difficulty": "medium",
                    "points": 5,
                    "topic": "Sequence & Order",
                    "source_context": current['text'] + " " + next_item['text'],
                    "confidence": 0.81,
                    "keywords": []
                })
        
        return questions
    
    # ========== HELPER FUNCTIONS FOR NEW QUESTION TYPES ==========
    
    def _create_inference(self, text: str) -> Optional[str]:
        """Create an inference statement from text"""
        text_lower = text.lower()
        
        # Pattern-based inference creation
        if 'advantage' in text_lower or 'benefit' in text_lower:
            return "This provides significant value in practical applications"
        elif 'enables' in text_lower or 'facilitates' in text_lower:
            return "This capability enhances overall system effectiveness"
        elif 'improves' in text_lower or 'enhances' in text_lower:
            return "This leads to better outcomes and efficiency"
        elif 'challenge' in text_lower or 'difficulty' in text_lower:
            return "This presents obstacles that require careful consideration"
        elif 'requires' in text_lower or 'needs' in text_lower:
            return "This necessitates specific resources or conditions"
        
        return None
    
    def _generate_inference_distractors(self, correct: str, kb: Dict) -> List[str]:
        """Generate plausible but incorrect inferences"""
        distractors = [
            "This has no practical implications",
            "This contradicts fundamental principles",
            "This is primarily theoretical with no real-world application",
            "This is outdated and no longer relevant"
        ]
        
        return random.sample(distractors, min(3, len(distractors)))
    
    def _create_application_scenario(self, term: str, definition: str) -> Optional[str]:
        """Create a realistic application scenario"""
        term_lower = term.lower()
        
        if 'data' in term_lower:
            return "When analyzing large datasets to extract business insights"
        elif 'model' in term_lower or 'algorithm' in term_lower:
            return "When building predictive systems for decision-making"
        elif 'security' in term_lower:
            return "When protecting sensitive information from unauthorized access"
        elif 'engineer' in term_lower:
            return "When building scalable data infrastructure"
        elif 'analyst' in term_lower:
            return "When interpreting data to support business decisions"
        elif 'process' in term_lower:
            return "When transforming raw information into actionable insights"
        elif 'tool' in term_lower or 'software' in term_lower:
            return "When automating repetitive data tasks"
        else:
            return f"When working with {term_lower} in professional contexts"
    
    def _create_distractor_scenarios(self, term: str, kb: Dict) -> List[str]:
        """Create distractor application scenarios"""
        distractors = [
            "When writing basic documentation for end users",
            "When performing routine system maintenance checks",
            "When organizing physical file storage systems",
            "When conducting general office administration tasks"
        ]
        
        return distractors
    
    def _extract_process_context(self, text: str) -> str:
        """Extract when a process would be used"""
        if 'analysis' in text.lower() or 'analyze' in text.lower():
            return "When examining data to discover patterns and insights"
        elif 'clean' in text.lower() or 'preprocess' in text.lower():
            return "When preparing raw data for analysis"
        elif 'model' in text.lower():
            return "When building predictive or analytical models"
        elif 'visualiz' in text.lower():
            return "When presenting data insights to stakeholders"
        else:
            return "When working through a structured data workflow"
    
    def _shorten_text(self, text: str, max_length: int) -> str:
        """Shorten text to max length"""
        if len(text) <= max_length:
            return text
        
        # Try to break at word boundary
        shortened = text[:max_length]
        last_space = shortened.rfind(' ')
        
        if last_space > max_length * 0.7:
            shortened = shortened[:last_space]
        
        return shortened + "..."
    
    # ========== HELPER FUNCTIONS ==========
    
    def _generate_smart_distractors(self, correct_answer: str, kb: Dict, num: int) -> List[str]:
        """Generate semantically plausible distractors"""
        distractors = []
        
        # Strategy 1: Use other definitions
        for def_info in kb['definitions']:
            if def_info['definition'] != correct_answer:
                distractors.append(def_info['definition'])
        
        # Strategy 2: Use related sentences
        for sent_info in kb['clean_sentences']:
            if sent_info['text'] != correct_answer and len(sent_info['text'].split()) > 8:
                # Extract the main clause
                main_clause = self._extract_main_clause(sent_info['text'])
                if main_clause and main_clause not in distractors:
                    distractors.append(main_clause)
        
        # Shuffle and select most different ones
        random.shuffle(distractors)
        
        # Use embeddings to select semantically different options
        if len(distractors) > num:
            all_options = [correct_answer] + distractors
            embeddings = self.sentence_model.encode(all_options)
            
            selected = []
            correct_emb = embeddings[0]
            
            for i in range(1, len(embeddings)):
                similarity = np.dot(correct_emb, embeddings[i])
                selected.append((distractors[i-1], similarity))
            
            # Sort by similarity (lower is more different)
            selected.sort(key=lambda x: x[1])
            distractors = [s[0] for s in selected[:num]]
        
        return distractors[:num]
    
    def _format_as_option(self, text: str) -> str:
        """Format text as a clean option"""
        # Remove leading/trailing punctuation
        text = text.strip().strip('.,;:')
        
        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        return text
    
    def _extract_action_from_sentence(self, sentence: str, person: str) -> str:
        """Extract what a person did from a sentence"""
        # Simple extraction: get text after the person's name
        if person in sentence:
            parts = sentence.split(person, 1)
            if len(parts) > 1:
                action = parts[1].strip()
                # Clean up
                action = re.sub(r'^[,\s]+', '', action)
                action = re.sub(r'[.!?]+$', '', action)
                return action[:80]  # Limit length
        return "contributed to this field"
    
    def _extract_event_from_sentence(self, sentence: str, date: str) -> Optional[str]:
        """Extract event description from sentence"""
        # Find what happened around the date
        if date in sentence:
            # Get verb phrases around the date
            doc = self.nlp(sentence)
            
            for token in doc:
                if token.pos_ == 'VERB':
                    # Get the verb phrase
                    phrase = ' '.join([t.text for t in token.subtree])
                    if len(phrase.split()) > 2 and len(phrase) < 80:
                        return phrase.lower()
        
        return None
    
    def _extract_main_clause(self, sentence: str) -> Optional[str]:
        """Extract main clause from a sentence"""
        doc = self.nlp(sentence)
        
        # Find root verb
        root = [token for token in doc if token.dep_ == 'ROOT']
        
        if root:
            root_token = root[0]
            # Get subtree
            clause = ' '.join([t.text for t in root_token.subtree])
            if 5 <= len(clause.split()) <= 25:
                return clause
        
        return None
    
    def _paraphrase_sentence(self, sentence: str, doc) -> Optional[str]:
        """Simple paraphrasing by restructuring"""
        # Strategy 1: Change active to passive or vice versa
        # Strategy 2: Reorder clauses
        # Strategy 3: Use synonyms for key verbs
        
        # For now, simple reordering
        if ',' in sentence:
            parts = sentence.split(',', 1)
            if len(parts) == 2 and len(parts[0].split()) > 3:
                # Sometimes reverse order
                if random.random() < 0.5:
                    paraphrased = parts[1].strip() + ', ' + parts[0].lower()
                    return paraphrased
        
        # Keep original if can't paraphrase well
        return None
    
    def _create_intelligent_false_statement(self, text: str, doc, kb: Dict) -> Optional[str]:
        """Create a false version intelligently"""
        
        # Strategy 1: Change numbers
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            num = numbers[0]
            try:
                new_num = str(int(num) + random.randint(10, 50))
                return text.replace(num, new_num, 1)
            except:
                pass
        
        # Strategy 2: Swap entities
        entities = [ent for ent in doc.ents if len(ent.text) > 2]
        if entities:
            target_ent = entities[0]
            for sent_info in kb['clean_sentences']:
                for other_ent in sent_info['doc'].ents:
                    if other_ent.text != target_ent.text and other_ent.label_ == target_ent.label_:
                        if target_ent.text in text:
                            return text.replace(target_ent.text, other_ent.text, 1)
        
        # Strategy 3: Negate
        if ' is ' in text and ' not ' not in text.lower():
            return text.replace(' is ', ' is not ', 1)
        if ' can ' in text and ' cannot ' not in text.lower():
            return text.replace(' can ', ' cannot ', 1)
        
        # Strategy 4: Swap key concepts
        for concept in kb['key_concepts'][:5]:
            if concept in text.lower():
                for other_concept in kb['key_concepts']:
                    if other_concept != concept and other_concept not in text.lower():
                        return text.replace(concept, other_concept, 1)
        
        return None
    
    def _group_related_concepts(self, kb: Dict) -> Dict[str, List[str]]:
        """Group related concepts together"""
        groups = defaultdict(list)
        
        # Group by common patterns
        for sent_info in kb['clean_sentences']:
            sent = sent_info['text']
            
            # Pattern: "X includes A, B, and C"
            match = re.search(r'([^:]+):\s*([^.]+)', sent)
            if match:
                category = match.group(1).strip()
                items = match.group(2).strip()
                
                # Split items
                item_list = re.split(r',\s*(?:and\s+)?', items)
                if len(item_list) >= 3:
                    groups[category] = [item.strip() for item in item_list]
        
        return groups
    
    def _find_false_item(self, true_items: List[str], kb: Dict) -> Optional[str]:
        """Find a plausible false item"""
        # Get all concepts
        all_concepts = kb['key_concepts']
        
        # Find one that's not in true_items
        for concept in all_concepts:
            if concept not in [item.lower() for item in true_items]:
                return concept.title()
        
        # Generic false items
        generic = [
            "Data visualization",
            "Cloud computing",
            "Network security",
            "Database management",
            "System administration"
        ]
        
        for g in generic:
            if g not in true_items:
                return g
        
        return None
    
    def _is_high_quality_question(self, q: Dict) -> bool:
        """STRICT validation - only pass HIGH QUALITY questions"""
        
        # Must have required fields
        if not all(k in q for k in ['text', 'correct_answer', 'type']):
            return False
        
        # Question text validation
        q_text = q['text']
        if not (20 <= len(q_text) <= 300):
            return False
        
        # No problematic characters
        if any(c in q_text for c in ['\\', '{', '}', '$']):
            return False
        
        # Must be readable
        alpha_ratio = sum(1 for c in q_text if c.isalpha() or c.isspace()) / max(len(q_text), 1)
        if alpha_ratio < 0.7:
            return False
        
        # Answer validation
        answer = q['correct_answer']
        if not answer or len(answer) < 2:
            return False
        
        # MCQ specific validation
        if q['type'] == 'multipleChoice':
            options = q.get('options', [])
            
            if not (3 <= len(options) <= 4):
                return False
            
            if len(set(options)) != len(options):
                return False
            
            if answer not in options:
                return False
            
            for opt in options:
                if not opt or len(opt) < 3 or len(opt) > 300:
                    return False
                if any(c in opt for c in ['\\', '{', '}', '$']):
                    return False
                opt_alpha_ratio = sum(1 for c in opt if c.isalpha() or c.isspace()) / max(len(opt), 1)
                if opt_alpha_ratio < 0.65:
                    return False
        
        # Confidence threshold
        if q.get('confidence', 0) < 0.70:
            return False
        
        return True
    
    def _select_diverse_questions(self, questions: List[Dict], target: int) -> List[Dict]:
        """Select diverse questions"""
        if len(questions) <= target:
            return questions
        
        # Prioritize by confidence and type diversity
        sorted_questions = sorted(questions, key=lambda x: x['confidence'], reverse=True)
        
        # Ensure type diversity
        type_counts = defaultdict(int)
        topic_counts = defaultdict(int)
        
        selected = []
        
        for q in sorted_questions:
            if len(selected) >= target:
                break
            
            q_type = q['type']
            q_topic = q.get('topic', 'General')
            
            # Prefer diverse types and topics
            if type_counts[q_type] < target // 2 and topic_counts[q_topic] < target // 3:
                selected.append(q)
                type_counts[q_type] += 1
                topic_counts[q_topic] += 1
        
        # Fill remaining with best questions
        for q in sorted_questions:
            if len(selected) >= target:
                break
            if q not in selected:
                selected.append(q)
        
        return selected
    
    def _set_difficulty(self, questions: List[Dict], target_diff: str) -> List[Dict]:
        """Set difficulty distribution"""
        if target_diff == "mixed":
            # Distribute: 30% easy, 50% medium, 20% hard
            easy_count = int(len(questions) * 0.3)
            medium_count = int(len(questions) * 0.5)
            
            for i, q in enumerate(questions):
                if i < easy_count:
                    q['difficulty'] = 'easy'
                    q['points'] = 3
                elif i < easy_count + medium_count:
                    q['difficulty'] = 'medium'
                    q['points'] = 5
                else:
                    q['difficulty'] = 'hard'
                    q['points'] = 7
        else:
            for q in questions:
                q['difficulty'] = target_diff
                q['points'] = {"easy": 3, "medium": 5, "hard": 7}.get(target_diff, 5)
        
        return questions