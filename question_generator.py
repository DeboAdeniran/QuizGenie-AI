import spacy
import nltk
from nltk.tokenize import sent_tokenize
import random
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict
import time
from loguru import logger
from sentence_transformers import SentenceTransformer
import numpy as np

class QuestionGenerator:
    """
    Advanced AI-powered question generation from text
    """
    
    def __init__(self, spacy_model: str = "en_core_web_lg"):
        """Initialize the question generator"""
        try:
            self.nlp = spacy.load(spacy_model)
            # Load sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("QuestionGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing QuestionGenerator: {e}")
            raise
        
        # Question templates
        self.mcq_templates = self._init_mcq_templates()
        self.tf_templates = self._init_tf_templates()
        
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
        """
        Generate questions from text
        """
        start_time = time.time()
        
        if question_types is None:
            question_types = ["multipleChoice", "trueFalse"]
        
        # Process text
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        logger.info(f"Processing {len(sentences)} sentences for question generation")
        
        # Extract key information
        entities = self._extract_entities(doc)
        facts = self._extract_facts(sentences)
        definitions = self._extract_definitions(sentences)
        
        # Generate questions
        questions = []
        
        # Calculate distribution
        distribution = self._calculate_distribution(
            num_questions, 
            question_types, 
            difficulty
        )
        
        # Generate each type of question
        for q_type, count in distribution['types'].items():
            if q_type == "multipleChoice":
                mcqs = self._generate_multiple_choice(
                    doc, sentences, entities, facts, count, distribution['difficulty']
                )
                questions.extend(mcqs)
            
            elif q_type == "trueFalse":
                tf_questions = self._generate_true_false(
                    sentences, facts, count, distribution['difficulty']
                )
                questions.extend(tf_questions)
            
            elif q_type == "shortAnswer":
                sa_questions = self._generate_short_answer(
                    sentences, definitions, entities, count, distribution['difficulty']
                )
                questions.extend(sa_questions)
        
        # Ensure context awareness
        if context_aware and len(questions) > 0:
            questions = self._ensure_context_diversity(questions, text)
        
        # Sort by difficulty and confidence
        questions = sorted(questions, key=lambda x: (-x['confidence'], x['difficulty']))
        
        # Limit to requested number
        questions = questions[:num_questions]
        
        # Calculate topics covered
        topics_covered = list(set([q.get('topic', 'General') for q in questions]))
        
        generation_time = time.time() - start_time
        
        return {
            "questions": questions,
            "generation_time": round(generation_time, 2),
            "total_questions": len(questions),
            "difficulty_distribution": {
                "easy": len([q for q in questions if q['difficulty'] == 'easy']),
                "medium": len([q for q in questions if q['difficulty'] == 'medium']),
                "hard": len([q for q in questions if q['difficulty'] == 'hard'])
            },
            "topics_covered": topics_covered
        }
    
    def _calculate_distribution(self, num_questions: int, question_types: List[str], 
                                difficulty: str) -> Dict[str, Any]:
        """Calculate question type and difficulty distribution"""
        # Type distribution
        type_dist = {}
        questions_per_type = num_questions // len(question_types)
        remainder = num_questions % len(question_types)
        
        for i, q_type in enumerate(question_types):
            count = questions_per_type + (1 if i < remainder else 0)
            type_dist[q_type] = count
        
        # Difficulty distribution
        if difficulty == "mixed":
            diff_dist = {
                "easy": num_questions // 3,
                "medium": num_questions // 3,
                "hard": num_questions - (2 * (num_questions // 3))
            }
        else:
            diff_dist = {difficulty: num_questions}
        
        return {"types": type_dist, "difficulty": diff_dist}
    
    def _extract_entities(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """Extract named entities with context"""
        entities = []
        for ent in doc.ents:
            if len(ent.text) > 2:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "context": ent.sent.text,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        return entities
    
    def _extract_facts(self, sentences: List[spacy.tokens.Span]) -> List[Dict[str, Any]]:
        """Extract factual statements"""
        facts = []
        
        for sent in sentences:
            # Look for sentences with strong subject-verb-object structure
            has_subj = any(token.dep_ == "nsubj" for token in sent)
            has_verb = any(token.pos_ == "VERB" for token in sent)
            has_obj = any(token.dep_ in ["dobj", "attr", "acomp"] for token in sent)
            
            if has_subj and has_verb and (has_obj or len(sent) > 5):
                facts.append({
                    "text": sent.text.strip(),
                    "root": sent.root.text,
                    "length": len(sent)
                })
        
        return facts
    
    def _extract_definitions(self, sentences: List[spacy.tokens.Span]) -> List[Dict[str, Any]]:
        """Extract definitions and explanatory sentences"""
        definitions = []
        
        definition_patterns = [
            r'\bis\s+(a|an|the)\s+',
            r'\brefers\s+to\s+',
            r'\bmeans\s+',
            r'\bdefined\s+as\s+',
            r'\bknown\s+as\s+'
        ]
        
        for sent in sentences:
            sent_text = sent.text.lower()
            for pattern in definition_patterns:
                if re.search(pattern, sent_text):
                    definitions.append({
                        "text": sent.text.strip(),
                        "subject": sent[:5].text  # Approximate subject
                    })
                    break
        
        return definitions
    
    def _generate_multiple_choice(self, doc, sentences, entities, facts, 
                                   count: int, difficulty_dist: Dict[str, int]) -> List[Dict]:
        """Generate multiple choice questions"""
        questions = []
        used_contexts = set()
        
        difficulties = []
        for diff, num in difficulty_dist.items():
            difficulties.extend([diff] * num)
        random.shuffle(difficulties)
        
        # Generate from entities
        for i, entity in enumerate(entities[:count]):
            if entity['context'] in used_contexts:
                continue
            
            difficulty = difficulties[i % len(difficulties)] if difficulties else "medium"
            
            question = self._create_entity_mcq(entity, doc, difficulty)
            if question:
                questions.append(question)
                used_contexts.add(entity['context'])
            
            if len(questions) >= count:
                break
        
        # Fill remaining with fact-based questions
        if len(questions) < count:
            for fact in facts:
                if fact['text'] in used_contexts or len(questions) >= count:
                    continue
                
                difficulty = difficulties[len(questions) % len(difficulties)] if difficulties else "medium"
                question = self._create_fact_mcq(fact, doc, difficulty)
                
                if question:
                    questions.append(question)
                    used_contexts.add(fact['text'])
        
        return questions
    
    def _create_entity_mcq(self, entity: Dict, doc, difficulty: str) -> Optional[Dict]:
        """Create MCQ from entity"""
        context = entity['context']
        entity_text = entity['text']
        entity_label = entity['label']
        
        # Generate question
        if entity_label == "PERSON":
            question_text = f"Who is {entity_text} in the context discussed?"
        elif entity_label in ["ORG", "PRODUCT"]:
            question_text = f"What is {entity_text}?"
        elif entity_label == "DATE":
            question_text = f"When did the event involving {entity_text} occur?"
        elif entity_label == "GPE":
            question_text = f"Where is {entity_text} located?"
        else:
            question_text = f"What role does {entity_text} play in the context?"
        
        # Generate options
        correct_answer = self._extract_answer_from_context(context, entity_text)
        if not correct_answer or len(correct_answer) < 5:
            return None
        
        distractors = self._generate_distractors(entity_text, entity_label, doc, 3)
        
        if len(distractors) < 3:
            return None
        
        options = [correct_answer] + distractors
        random.shuffle(options)
        
        # Determine points based on difficulty
        points = {"easy": 3, "medium": 5, "hard": 7}.get(difficulty, 5)
        
        return {
            "type": "multipleChoice",
            "text": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": f"According to the text: {context[:200]}...",
            "difficulty": difficulty,
            "points": points,
            "topic": entity_label.title(),
            "source_context": context,
            "confidence": 0.85,
            "keywords": [entity_text]
        }
    
    def _create_fact_mcq(self, fact: Dict, doc, difficulty: str) -> Optional[Dict]:
        """Create MCQ from factual statement"""
        fact_text = fact['text']
        sent = self.nlp(fact_text)
        
        # Find the main verb and subject
        root = None
        subject = None
        
        for token in sent:
            if token.dep_ == "ROOT":
                root = token
            if token.dep_ == "nsubj":
                subject = token
        
        if not root or not subject:
            return None
        
        # Generate question by blanking the object/complement
        question_text = f"What {root.lemma_} {subject.text}?"
        
        # Extract answer
        obj_tokens = [token for token in sent if token.dep_ in ["dobj", "attr", "acomp"]]
        if not obj_tokens:
            return None
        
        correct_answer = " ".join([t.text for t in obj_tokens])
        
        # Generate distractors
        distractors = self._generate_semantic_distractors(correct_answer, doc, 3)
        
        if len(distractors) < 2:
            return None
        
        options = [correct_answer] + distractors
        random.shuffle(options)
        
        points = {"easy": 3, "medium": 5, "hard": 7}.get(difficulty, 5)
        
        return {
            "type": "multipleChoice",
            "text": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": f"Based on the statement: {fact_text}",
            "difficulty": difficulty,
            "points": points,
            "topic": "General",
            "source_context": fact_text,
            "confidence": 0.75,
            "keywords": [subject.text, root.text]
        }
    
    def _generate_distractors(self, correct: str, entity_type: str, 
                             doc, count: int) -> List[str]:
        """Generate plausible distractors"""
        distractors = []
        
        # Extract similar entities of the same type
        for ent in doc.ents:
            if ent.label_ == entity_type and ent.text != correct and len(ent.text) > 2:
                distractors.append(ent.text)
        
        # If not enough, generate generic ones
        if len(distractors) < count:
            generic = self._get_generic_distractors(entity_type, count - len(distractors))
            distractors.extend(generic)
        
        return distractors[:count]
    
    def _generate_semantic_distractors(self, correct: str, doc, count: int) -> List[str]:
        """Generate semantically similar distractors"""
        # Extract noun phrases from document
        candidates = []
        for chunk in doc.noun_chunks:
            if chunk.text != correct and 2 < len(chunk.text) < 50:
                candidates.append(chunk.text)
        
        if not candidates:
            return self._get_generic_distractors("GENERAL", count)
        
        # Use sentence transformer to find similar phrases
        if len(candidates) > 0:
            correct_embedding = self.sentence_model.encode([correct])[0]
            candidate_embeddings = self.sentence_model.encode(candidates)
            
            # Calculate similarities
            similarities = np.dot(candidate_embeddings, correct_embedding)
            top_indices = np.argsort(similarities)[-count:]
            
            return [candidates[i] for i in top_indices if candidates[i] != correct]
        
        return []
    
    def _get_generic_distractors(self, entity_type: str, count: int) -> List[str]:
        """Get generic distractors based on entity type"""
        generic_distractors = {
            "PERSON": ["John Smith", "Jane Doe", "Dr. Johnson", "Prof. Williams"],
            "ORG": ["ABC Corporation", "XYZ Institute", "Global Systems", "Tech Industries"],
            "GPE": ["New York", "London", "Tokyo", "Sydney"],
            "DATE": ["2020", "January 2021", "Last year", "Next month"],
            "GENERAL": ["Option A", "Alternative B", "Choice C", "Selection D"]
        }
        
        options = generic_distractors.get(entity_type, generic_distractors["GENERAL"])
        return random.sample(options, min(count, len(options)))
    
    def _extract_answer_from_context(self, context: str, entity: str) -> str:
        """Extract the actual answer from context"""
        # Find the sentence or clause containing the entity
        sent = self.nlp(context)
        
        for token in sent:
            if entity.lower() in token.text.lower():
                # Get the dependent phrase
                answer_parts = [token.text]
                for child in token.children:
                    if child.dep_ in ["amod", "compound", "prep"]:
                        answer_parts.extend([c.text for c in child.subtree])
                
                return " ".join(answer_parts)
        
        return entity
    
    def _generate_true_false(self, sentences, facts, count: int, 
                            difficulty_dist: Dict[str, int]) -> List[Dict]:
        """Generate True/False questions"""
        questions = []
        
        difficulties = []
        for diff, num in difficulty_dist.items():
            difficulties.extend([diff] * num)
        random.shuffle(difficulties)
        
        for i, fact in enumerate(facts[:count * 2]):  # Generate extra for filtering
            if len(questions) >= count:
                break
            
            difficulty = difficulties[i % len(difficulties)] if difficulties else "medium"
            
            # Randomly decide if true or false
            is_true = random.choice([True, False])
            
            if is_true:
                question_text = fact['text']
                correct_answer = "True"
                explanation = "This statement is directly stated in the text."
            else:
                # Modify the fact to make it false
                modified = self._create_false_statement(fact['text'])
                if not modified:
                    continue
                question_text = modified
                correct_answer = "False"
                explanation = f"This is incorrect. The actual statement is: {fact['text'][:100]}..."
            
            points = {"easy": 2, "medium": 3, "hard": 4}.get(difficulty, 3)
            
            questions.append({
                "type": "trueFalse",
                "text": question_text,
                "options": ["True", "False"],
                "correct_answer": correct_answer,
                "explanation": explanation,
                "difficulty": difficulty,
                "points": points,
                "topic": "General",
                "source_context": fact['text'],
                "confidence": 0.80,
                "keywords": [fact['root']]
            })
        
        return questions
    
    def _create_false_statement(self, true_statement: str) -> Optional[str]:
        """Modify a statement to make it false"""
        doc = self.nlp(true_statement)
        
        # Strategy 1: Negate the main verb
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                # Add negation
                if any(child.dep_ == "neg" for child in token.children):
                    # Already negative, make positive
                    false_statement = true_statement.replace(" not ", " ").replace(" n't ", " ")
                else:
                    # Make negative
                    false_statement = true_statement.replace(token.text, f"does not {token.text}")
                
                return false_statement
        
        # Strategy 2: Replace a key entity/number
        entities = [ent for ent in doc.ents if ent.label_ in ["CARDINAL", "QUANTITY", "PERCENT"]]
        if entities:
            ent = random.choice(entities)
            try:
                num = int(re.findall(r'\d+', ent.text)[0])
                new_num = num + random.randint(10, 50)
                return true_statement.replace(ent.text, str(new_num))
            except:
                pass
        
        return None
    
    def _generate_short_answer(self, sentences, definitions, entities, 
                               count: int, difficulty_dist: Dict[str, int]) -> List[Dict]:
        """Generate short answer questions"""
        questions = []
        
        # Use definitions for short answer questions
        for i, definition in enumerate(definitions[:count]):
            if len(questions) >= count:
                break
            
            # Extract the term being defined
            sent = self.nlp(definition['text'])
            subject = definition['subject']
            
            question_text = f"What is {subject}?"
            correct_answer = definition['text'].split(' is ')[-1].strip('.')
            
            if len(correct_answer) > 10:
                difficulty = "medium"
                points = 5
                
                questions.append({
                    "type": "shortAnswer",
                    "text": question_text,
                    "options": None,
                    "correct_answer": correct_answer,
                    "explanation": f"Full definition: {definition['text']}",
                    "difficulty": difficulty,
                    "points": points,
                    "topic": "Definitions",
                    "source_context": definition['text'],
                    "confidence": 0.70,
                    "keywords": [subject]
                })
        
        return questions
    
    def _ensure_context_diversity(self, questions: List[Dict], text: str) -> List[Dict]:
        """Ensure questions cover different parts of the text"""
        if len(questions) <= 1:
            return questions
        
        # Calculate embeddings for source contexts
        contexts = [q['source_context'] for q in questions]
        embeddings = self.sentence_model.encode(contexts)
        
        # Select diverse questions
        selected = [0]  # Always include the first
        
        for i in range(1, len(questions)):
            # Calculate min similarity to already selected
            min_similarity = min([
                np.dot(embeddings[i], embeddings[j]) 
                for j in selected
            ])
            
            # If sufficiently different, include it
            if min_similarity < 0.7 or len(selected) < len(questions) // 2:
                selected.append(i)
        
        return [questions[i] for i in selected]
    
    def _init_mcq_templates(self) -> List[str]:
        """Initialize MCQ question templates"""
        return [
            "What is {concept}?",
            "Which of the following best describes {concept}?",
            "According to the text, {concept} refers to:",
            "What is the primary function of {concept}?",
            "Which statement about {concept} is correct?"
        ]
    
    def _init_tf_templates(self) -> List[str]:
        """Initialize True/False templates"""
        return [
            "{statement}",
            "True or False: {statement}",
            "Is the following statement correct? {statement}"
        ]
    
    def validate_question(self, question: Dict) -> Dict[str, Any]:
        """Validate question quality"""
        issues = []
        score = 100
        
        # Check question text length
        if len(question['text']) < 10:
            issues.append("Question text too short")
            score -= 20
        
        if len(question['text']) > 300:
            issues.append("Question text too long")
            score -= 10
        
        # Check options for MCQ
        if question['type'] == 'multipleChoice':
            if not question.get('options') or len(question['options']) < 2:
                issues.append("Insufficient options")
                score -= 30
            
            # Check for duplicate options
            if len(question['options']) != len(set(question['options'])):
                issues.append("Duplicate options found")
                score -= 20
        
        # Check if correct answer exists
        if question['type'] in ['multipleChoice', 'trueFalse']:
            if question['correct_answer'] not in question['options']:
                issues.append("Correct answer not in options")
                score -= 50
        
        return {
            "is_valid": score >= 50,
            "quality_score": max(0, score),
            "issues": issues
        }