import anthropic
import json
import os
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests
from datetime import datetime

# Define data structures for our research agent
@dataclass
class ResearchDocument:
    """Represents a research document or paper."""
    id: str
    title: str
    authors: List[str]
    publication_date: str
    source: str
    content: str
    abstract: str
    url: Optional[str] = None
    citation: Optional[str] = None
    relevance_score: Optional[float] = None

@dataclass
class ResearchTopic:
    """Represents a research topic or question."""
    id: str
    query: str
    subtopics: List[str]
    keywords: List[str]
    scope: Dict[str, Any]  # time range, domains, etc.

@dataclass
class ResearchFinding:
    """Represents an extracted finding or insight."""
    id: str
    content: str
    source_documents: List[str]  # document IDs
    confidence: float
    tags: List[str]

@dataclass
class ResearchSynthesis:
    """Represents a synthesized research output."""
    id: str
    title: str
    summary: str
    key_findings: List[ResearchFinding]
    document_coverage: List[str]  # document IDs
    gaps_identified: List[str]
    future_directions: List[str]
    bibliography: List[str]

class DeepResearchAgent:
    """
    A comprehensive research agent powered by Claude 3.7 that can perform
    deep research on academic and professional topics.
    """
    
    def __init__(self, api_key: str, search_api_key: str = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.search_api_key = search_api_key
        self.documents = {}  # Store retrieved documents
        self.findings = {}   # Store extracted findings
        self.research_plans = {}  # Store research plans
        self.syntheses = {}  # Store final syntheses
        
    def conduct_research(self, query: str, depth: str = "comprehensive", 
                         time_limit: int = None) -> Dict[str, Any]:
        """
        Main entry point for conducting research on a given query.
        
        Args:
            query: The research question or topic
            depth: "quick", "standard", or "comprehensive"
            time_limit: Maximum time in seconds to spend on research
            
        Returns:
            A dictionary containing the research results
        """
        try:
            start_time = time.time()
            
            # 1. Create a research plan
            print("Creating research plan...")
            research_topic = self._create_research_plan(query, depth)
            
            # 2. Retrieve relevant documents
            print("Retrieving documents...")
            retrieved_docs = self._retrieve_documents(research_topic)
            
            # 3. Extract key information from documents
            print("Extracting information...")
            findings = self._extract_information(retrieved_docs, research_topic)
            
            # 4. Synthesize information
            print("Synthesizing findings...")
            synthesis = self._synthesize_findings(findings, research_topic)
            
            # 5. Generate insights and identify gaps
            print("Generating insights...")
            enhanced_synthesis = self._generate_insights(synthesis, research_topic)
            
            # 6. Format and finalize the research output
            print("Formatting research output...")
            final_output = self._format_research_output(enhanced_synthesis)
            
            research_time = time.time() - start_time
            print(f"Research completed in {research_time:.2f} seconds")
            
            return final_output
        except Exception as e:
            print(f"Error during research: {str(e)}")
            raise
        
    def _create_research_plan(self, query: str, depth: str) -> ResearchTopic:
        """
        Use Claude 3.7 to create a detailed research plan for the query.
        """
        try:
            planning_prompt = f"""
            Create a detailed research plan for the following query:
            
            QUERY: {query}
            DEPTH: {depth}
            
            Your task is to:
            1. Break down this research question into clear subtopics
            2. Identify key search terms and keywords for each subtopic
            3. Define the scope of research (time period, domains, etc.)
            4. Identify potential sources of information
            5. Suggest a structured approach to investigate this topic
            
            Format your response as a valid JSON object with the following structure:
            {{
                "main_query": "The main research question",
                "subtopics": ["subtopic1", "subtopic2", ...],
                "keywords": ["keyword1", "keyword2", ...],
                "scope": {{
                    "time_range": ["start_year", "end_year"],
                    "domains": ["domain1", "domain2", ...],
                    "excluded_areas": ["area1", "area2", ...]
                }},
                "potential_sources": ["source1", "source2", ...],
                "research_approach": "Description of the approach"
            }}
            """
            
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=2000,
                system="You are a research planning assistant that creates comprehensive research plans.",
                messages=[{"role": "user", "content": planning_prompt}]
            )
            
            # Extract the content from the response
            response_content = response.content[0].text
            
            # Extract JSON from the response
            plan_json = self._extract_json(response_content)
            
            # Default values in case parsing fails
            default_scope = {
                "time_range": ["2020", "2023"],
                "domains": ["General"],
                "excluded_areas": []
            }
            
            # Create a ResearchTopic object with fallbacks for missing data
            topic_id = f"topic_{int(time.time())}"
            research_topic = ResearchTopic(
                id=topic_id,
                query=plan_json.get("main_query", query),
                subtopics=plan_json.get("subtopics", [query.split(" and ")[0], query.split(" and ")[-1]]),
                keywords=plan_json.get("keywords", query.split()),
                scope=plan_json.get("scope", default_scope)
            )
            
            # Store the plan
            self.research_plans[topic_id] = {
                "plan": plan_json,
                "created_at": datetime.now().isoformat(),
                "research_topic": research_topic
            }
            
            return research_topic
        except Exception as e:
            print(f"Error creating research plan: {str(e)}")
            # Create a minimal research topic as fallback
            topic_id = f"topic_{int(time.time())}"
            fallback_topic = ResearchTopic(
                id=topic_id,
                query=query,
                subtopics=[query],
                keywords=query.split(),
                scope={"time_range": ["2020", "2023"], "domains": ["General"], "excluded_areas": []}
            )
            return fallback_topic
    
    def _retrieve_documents(self, research_topic: ResearchTopic) -> List[ResearchDocument]:
        """
        Retrieve relevant documents based on the research plan.
        This simulates document retrieval with a mock function.
        """
        try:
            retrieved_documents = []
            
            if self.search_api_key:
                # Example of using a real search API
                documents = self._search_academic_databases(research_topic)
                retrieved_documents.extend(documents)
            else:
                # For demo purposes without an API key, generate mock documents
                mock_documents = self._generate_mock_documents(research_topic)
                retrieved_documents.extend(mock_documents)
            
            # Store documents for later use
            for doc in retrieved_documents:
                self.documents[doc.id] = doc
                
            return retrieved_documents
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            # Return empty list as fallback
            return []
    
    def _search_academic_databases(self, research_topic: ResearchTopic) -> List[ResearchDocument]:
        """
        Search academic databases using API calls.
        This is a mock implementation.
        """
        try:
            documents = []
            
            # Using only the first 3 keywords for demo purposes
            for keyword in research_topic.keywords[:3]:
                # Use mock data instead of real API calls
                mock_results = self._mock_api_results(keyword)
                
                for result in mock_results:
                    doc_id = f"doc_{result['id']}"
                    document = ResearchDocument(
                        id=doc_id,
                        title=result["title"],
                        authors=result["authors"],
                        publication_date=result["date"],
                        source=result["journal"],
                        content=result["abstract"],
                        abstract=result["abstract"],
                        url=result["url"],
                        relevance_score=result["score"]
                    )
                    documents.append(document)
            
            return documents
        except Exception as e:
            print(f"Error searching academic databases: {str(e)}")
            return []
    
    # Update the mock data generator to use more recent dates
    def _mock_api_results(self, keyword: str) -> List[Dict]:
        """Generate mock API results with more recent dates."""
        import random
        
        journals = ["Nature", "Science", "PNAS", "Cell", "The Lancet"]
        
        results = []
        for i in range(3):  # Generate 3 mock results per keyword
            results.append({
                "id": f"{int(time.time())}{i}",
                "title": f"Research on {keyword.title()}: Advances and Implications",
                "authors": ["J. Smith", "A. Johnson", "M. Williams"],
                "date": f"202{random.randint(3, 4)}-{random.randint(1, 12):02d}",  # 2023-2024 dates
                "journal": random.choice(journals),
                "abstract": f"This paper investigates the latest developments in {keyword} from 2023-2024. "
                        f"We found significant results related to {keyword} that have "
                        f"implications for future research and applications.",
                "url": f"https://example.com/papers/{keyword.replace(' ', '_')}{i}",
                "score": random.uniform(0.7, 0.99)
            })
        
        return results
    
    def _generate_mock_documents(self, research_topic: ResearchTopic) -> List[ResearchDocument]:
        """Generate mock documents for demonstration purposes."""
        try:
            documents = []
            
            # Generate 1-2 documents for each subtopic
            for i, subtopic in enumerate(research_topic.subtopics):
                for j in range(2):
                    doc_id = f"doc_{int(time.time())}_{i}_{j}"
                    
                    # Safely access keywords
                    keyword_idx = min(i, len(research_topic.keywords)-1) if research_topic.keywords else 0
                    keyword_idx2 = min(i+1, len(research_topic.keywords)-1) if research_topic.keywords else 0
                    keyword_idx3 = min(i+2, len(research_topic.keywords)-1) if research_topic.keywords else 0
                    
                    keyword = research_topic.keywords[keyword_idx] if research_topic.keywords else subtopic
                    keyword2 = research_topic.keywords[keyword_idx2] if research_topic.keywords else subtopic
                    keyword3 = research_topic.keywords[keyword_idx3] if research_topic.keywords else subtopic
                    
                    # Safely access time range
                    time_range = research_topic.scope.get('time_range', ['2020', '2023'])
                    
                    # Create more realistic mock content
                    content = f"""
                    Abstract:
                    This research explores {subtopic} in detail, with a focus on recent developments.
                    We analyze the implications of {keyword} on various aspects of {subtopic}.
                    
                    Introduction:
                    The field of {subtopic} has seen significant advancement in recent years.
                    This paper aims to provide a comprehensive overview of the current state of research
                    and identify future directions. We begin by examining the fundamental concepts
                    related to {keyword}.
                    
                    Methodology:
                    Our research methodology involved a systematic review of literature published
                    between {time_range[0]} and {time_range[1]}.
                    We analyzed over 50 papers from leading journals in the field.
                    
                    Findings:
                    Our analysis revealed several key patterns. First, {subtopic} is increasingly
                    recognized as critical in understanding {research_topic.query}.
                    Second, the connection between {subtopic} and {keyword2}
                    appears stronger than previously thought.
                    
                    Discussion:
                    These findings suggest that future research should focus on exploring the
                    relationship between {subtopic} and {keyword3}.
                    There are still significant gaps in our understanding of how {subtopic}
                    affects practical applications in real-world scenarios.
                    
                    Conclusion:
                    This research contributes to the growing body of knowledge on {research_topic.query}
                    and highlights the importance of {subtopic} in this context.
                    """
                    
                    # Create the document
                    document = ResearchDocument(
                        id=doc_id,
                        title=f"Analysis of {subtopic.title()} in the Context of {research_topic.query}",
                        authors=[f"Author {j+1}", f"Author {j+2}", f"Author {j+3}"],
                        publication_date=f"202{j}-{(i+1)%12+1:02d}",
                        source=f"Journal of {subtopic.title()} Research",
                        content=content,
                        abstract=content.split("Abstract:")[1].split("Introduction:")[0].strip() if "Abstract:" in content and "Introduction:" in content else content[:200],
                        url=f"https://example.org/papers/{doc_id}",
                        relevance_score=0.8 + (j * 0.1)
                    )
                    
                    documents.append(document)
            
            return documents
        except Exception as e:
            print(f"Error generating mock documents: {str(e)}")
            # Create a single fallback document
            fallback_id = f"doc_fallback_{int(time.time())}"
            fallback_doc = ResearchDocument(
                id=fallback_id,
                title=f"Overview of {research_topic.query}",
                authors=["Default Author"],
                publication_date="2023-01-01",
                source="Default Source",
                content=f"This is a default document about {research_topic.query}.",
                abstract=f"Default abstract about {research_topic.query}."
            )
            return [fallback_doc]
    
    def _extract_information(self, documents: List[ResearchDocument], 
                            research_topic: ResearchTopic) -> List[ResearchFinding]:
        """
        Use Claude 3.7 to extract key information and findings from documents.
        """
        try:
            findings = []
            
            if not documents:
                print("Warning: No documents to extract information from")
                return []
            
            # Process each document to extract key findings
            for document in documents:
                extraction_prompt = f"""
                Extract key information and findings from the following research document.
                
                DOCUMENT TITLE: {document.title}
                AUTHORS: {', '.join(document.authors)}
                PUBLICATION: {document.source} ({document.publication_date})
                
                CONTENT:
                {document.content[:8000]}  # Limit content length
                
                RESEARCH QUESTION:
                {research_topic.query}
                
                RELEVANT SUBTOPICS:
                {', '.join(research_topic.subtopics)}
                
                Extract the following:
                1. Key findings or claims made in the document
                2. Evidence provided to support these findings
                3. Methodologies used
                4. Limitations mentioned
                5. How the findings relate to our research question
                
                Format your response as a valid JSON object with the following structure:
                {{
                    "key_findings": [
                        {{
                            "finding": "Description of finding",
                            "evidence": "Evidence provided",
                            "relevance": "Relevance to research question",
                            "confidence": 0.95  # Value between 0-1 representing confidence in the extraction
                        }},
                        ...
                    ],
                    "methodologies": ["methodology1", "methodology2", ...],
                    "limitations": ["limitation1", "limitation2", ...],
                    "research_gaps": ["gap1", "gap2", ...]
                }}
                """
                
                response = self.client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=2500,
                    system="You are a research analysis assistant that extracts key information from academic papers.",
                    messages=[{"role": "user", "content": extraction_prompt}]
                )
                
                # Extract the content from the response
                response_content = response.content[0].text
                
                # Extract the JSON from the response
                findings_json = self._extract_json(response_content)
                
                # Default if key_findings is missing
                key_findings = findings_json.get("key_findings", [])
                if not key_findings:
                    key_findings = [{
                        "finding": f"The document discusses {research_topic.query}.",
                        "evidence": "Not specified",
                        "relevance": "Directly related to the research question",
                        "confidence": 0.7
                    }]
                
                # Create ResearchFinding objects for each key finding
                for i, finding_data in enumerate(key_findings):
                    finding_id = f"finding_{document.id}_{i}"
                    finding = ResearchFinding(
                        id=finding_id,
                        content=finding_data.get("finding", "No finding specified"),
                        source_documents=[document.id],
                        confidence=finding_data.get("confidence", 0.7),
                        tags=[research_topic.query] + [st for st in research_topic.subtopics 
                                                     if st.lower() in finding_data.get("finding", "").lower()]
                    )
                    findings.append(finding)
                    self.findings[finding_id] = finding
                    
            return findings
        except Exception as e:
            print(f"Error extracting information: {str(e)}")
            # Create a fallback finding
            fallback_id = f"finding_fallback_{int(time.time())}"
            fallback_finding = ResearchFinding(
                id=fallback_id,
                content=f"General information about {research_topic.query}",
                source_documents=[doc.id for doc in documents[:1]] if documents else [],
                confidence=0.5,
                tags=[research_topic.query]
            )
            return [fallback_finding]
    
    def _synthesize_findings(self, findings: List[ResearchFinding], 
                            research_topic: ResearchTopic) -> ResearchSynthesis:
        """
        Use Claude 3.7 to synthesize findings into a coherent narrative.
        """
        try:
            if not findings:
                print("Warning: No findings to synthesize")
                return self._create_default_synthesis(research_topic)
            
            # Prepare findings data for Claude
            findings_data = []
            for finding in findings:
                if not finding.source_documents:
                    continue
                    
                doc_id = finding.source_documents[0]
                if doc_id not in self.documents:
                    continue
                    
                document = self.documents[doc_id]
                findings_data.append({
                    "finding": finding.content,
                    "source": document.title,
                    "authors": ", ".join(document.authors),
                    "publication_date": document.publication_date,
                    "confidence": finding.confidence
                })
            
            if not findings_data:
                print("Warning: No valid findings data for synthesis")
                return self._create_default_synthesis(research_topic)
            
            synthesis_prompt = f"""
            Synthesize the following research findings into a coherent narrative.
            
            RESEARCH QUESTION:
            {research_topic.query}
            
            SUBTOPICS:
            {', '.join(research_topic.subtopics)}
            
            FINDINGS:
            {json.dumps(findings_data, indent=2)}
            
            Your task is to:
            1. Create a comprehensive synthesis of these findings
            2. Identify patterns, trends, and connections across findings
            3. Note contradictions or inconsistencies between findings
            4. Identify gaps in the research
            5. Suggest potential directions for future research
            
            Format your response as a valid JSON object with the following structure:
            {{
                "title": "Title for the research synthesis",
                "executive_summary": "Brief summary of key insights",
                "synthesis_by_subtopic": [
                    {{
                        "subtopic": "Subtopic name",
                        "synthesis": "Detailed synthesis of findings for this subtopic",
                        "key_insights": ["insight1", "insight2", ...]
                    }},
                    ...
                ],
                "cross_cutting_themes": ["theme1", "theme2", ...],
                "contradictions": ["contradiction1", "contradiction2", ...],
                "research_gaps": ["gap1", "gap2", ...],
                "future_directions": ["direction1", "direction2", ...]
            }}
            """
            
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4000,
                system="You are a research synthesis assistant that integrates findings into coherent narratives.",
                messages=[{"role": "user", "content": synthesis_prompt}]
            )
            
            # Extract the content from the response
            response_content = response.content[0].text
            
            # Extract the JSON from the response
            synthesis_json = self._extract_json(response_content)
            
            # Default values if keys are missing
            synthesis_by_subtopic = synthesis_json.get("synthesis_by_subtopic", [])
            if not synthesis_by_subtopic and research_topic.subtopics:
                synthesis_by_subtopic = [{
                    "subtopic": subtopic,
                    "synthesis": f"Analysis of {subtopic} in relation to {research_topic.query}.",
                    "key_insights": [f"Key insight about {subtopic}."]
                } for subtopic in research_topic.subtopics]
            
            # Create the ResearchSynthesis
            synthesis_id = f"synthesis_{int(time.time())}"
            document_ids = list(set([doc_id for finding in findings 
                                    for doc_id in finding.source_documents if doc_id in self.documents]))
            
            key_findings = []
            for i, subtopic in enumerate(synthesis_by_subtopic):
                insights = subtopic.get("key_insights", [])
                if insights:
                    for j, insight in enumerate(insights):
                        finding_id = f"syn_finding_{i}_{j}"
                        finding = ResearchFinding(
                            id=finding_id,
                            content=insight,
                            source_documents=document_ids,
                            confidence=0.9,
                            tags=[subtopic.get("subtopic", "General")]
                        )
                        key_findings.append(finding)
            
            # If no key findings were created, add a default one
            if not key_findings:
                finding_id = f"syn_finding_default"
                finding = ResearchFinding(
                    id=finding_id,
                    content=f"General insight about {research_topic.query}.",
                    source_documents=document_ids,
                    confidence=0.7,
                    tags=["General"]
                )
                key_findings.append(finding)
            
            synthesis = ResearchSynthesis(
                id=synthesis_id,
                title=synthesis_json.get("title", f"Research on {research_topic.query}"),
                summary=synthesis_json.get("executive_summary", f"Summary of research on {research_topic.query}."),
                key_findings=key_findings,
                document_coverage=document_ids,
                gaps_identified=synthesis_json.get("research_gaps", [f"Further research needed on {research_topic.query}."]),
                future_directions=synthesis_json.get("future_directions", ["Expand the scope of research."]),
                bibliography=[self._format_citation(self.documents[doc_id]) for doc_id in document_ids if doc_id in self.documents]
            )
            
            self.syntheses[synthesis_id] = synthesis
            return synthesis
        except Exception as e:
            print(f"Error synthesizing findings: {str(e)}")
            return self._create_default_synthesis(research_topic)
    
    def _create_default_synthesis(self, research_topic: ResearchTopic) -> ResearchSynthesis:
        """Create a default synthesis when the normal process fails."""
        synthesis_id = f"synthesis_default_{int(time.time())}"
        
        default_finding = ResearchFinding(
            id=f"default_finding_{int(time.time())}",
            content=f"General information about {research_topic.query}.",
            source_documents=[],
            confidence=0.5,
            tags=["General"]
        )
        
        return ResearchSynthesis(
            id=synthesis_id,
            title=f"Research on {research_topic.query}",
            summary=f"This is a default synthesis for {research_topic.query}.",
            key_findings=[default_finding],
            document_coverage=[],
            gaps_identified=[f"More research needed on {research_topic.query}."],
            future_directions=["Expand the scope of research."],
            bibliography=[]
        )
    
    def _generate_insights(self, synthesis: ResearchSynthesis, 
                          research_topic: ResearchTopic) -> ResearchSynthesis:
        """
        Use Claude 3.7 to generate deeper insights and identify patterns.
        """
        try:
            # Prepare the synthesis data for Claude
            synthesis_data = {
                "title": synthesis.title,
                "summary": synthesis.summary,
                "key_findings": [finding.content for finding in synthesis.key_findings],
                "gaps": synthesis.gaps_identified,
                "future_directions": synthesis.future_directions
            }
            
            insight_prompt = f"""
            Analyze this research synthesis to generate deeper insights and identify patterns
            that may not be immediately obvious.
            
            RESEARCH QUESTION:
            {research_topic.query}
            
            RESEARCH SYNTHESIS:
            {json.dumps(synthesis_data, indent=2)}
            
            Your task is to:
            1. Identify non-obvious patterns or connections in the research
            2. Suggest potential paradigm shifts or transformative ideas
            3. Identify interdisciplinary connections
            4. Suggest practical applications of the research findings
            5. Provide a critical perspective on the limitations of current approaches
            
            Format your response as a valid JSON object with the following structure:
            {{
                "deeper_patterns": ["pattern1", "pattern2", ...],
                "transformative_ideas": ["idea1", "idea2", ...],
                "interdisciplinary_connections": ["connection1", "connection2", ...],
                "practical_applications": ["application1", "application2", ...],
                "critical_perspective": "Critical perspective on the research",
                "enhanced_future_directions": ["direction1", "direction2", ...]
            }}
            """
            
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=3000,
                system="You are a research insight generator that identifies deeper patterns and connections.",
                messages=[{"role": "user", "content": insight_prompt}]
            )
            
            # Extract the content from the response
            response_content = response.content[0].text
            
            # Extract the JSON from the response
            insights_json = self._extract_json(response_content)
            
            # Create a copy of the synthesis to enhance
            enhanced_synthesis = ResearchSynthesis(
                id=synthesis.id,
                title=synthesis.title,
                summary=synthesis.summary,
                key_findings=list(synthesis.key_findings),  # Create a new list with the same elements
                document_coverage=synthesis.document_coverage,
                gaps_identified=synthesis.gaps_identified,
                future_directions=insights_json.get("enhanced_future_directions", synthesis.future_directions),
                bibliography=synthesis.bibliography
            )
            
            # Add new findings for the patterns and applications
            for i, pattern in enumerate(insights_json.get("deeper_patterns", [])):
                finding_id = f"pattern_finding_{i}"
                finding = ResearchFinding(
                    id=finding_id,
                    content=pattern,
                    source_documents=synthesis.document_coverage,
                    confidence=0.85,
                    tags=["pattern", "insight"]
                )
                enhanced_synthesis.key_findings.append(finding)
                self.findings[finding_id] = finding
                
            for i, application in enumerate(insights_json.get("practical_applications", [])):
                finding_id = f"application_finding_{i}"
                finding = ResearchFinding(
                    id=finding_id,
                    content=application,
                    source_documents=synthesis.document_coverage,
                    confidence=0.8,
                    tags=["application", "insight"]
                )
                enhanced_synthesis.key_findings.append(finding)
                self.findings[finding_id] = finding
                
            return enhanced_synthesis
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
            # Return the original synthesis if there's an error
            return synthesis


    def _format_research_output(self, synthesis: ResearchSynthesis) -> Dict[str, Any]:
        """
        Format the final research output for presentation.
        """
        try:
            # Get all the documents referenced in the synthesis
            documents = [self.documents[doc_id] for doc_id in synthesis.document_coverage if doc_id in self.documents]
            
            # Prepare the data for Claude to format
            format_data = {
                "title": synthesis.title,
                "summary": synthesis.summary,
                "key_findings": [finding.content for finding in synthesis.key_findings],
                "future_directions": synthesis.future_directions,
                "gaps": synthesis.gaps_identified,
                "sources": [
                    {
                        "title": doc.title,
                        "authors": doc.authors,
                        "publication": doc.source,
                        "date": doc.publication_date,
                        "url": doc.url
                    }
                    for doc in documents
                ]
            }
            
            formatting_prompt = f"""
            Format this research synthesis into a professional research report.
            
            RESEARCH DATA:
            {json.dumps(format_data, indent=2)}
            
            Create a comprehensive research report with the following sections:
            1. Executive Summary
            2. Introduction and Background
            3. Methodology
            4. Key Findings
            5. Discussion and Implications
            6. Research Gaps
            7. Future Directions
            8. Conclusion
            9. Bibliography
            
            Format your response as a valid JSON object that contains the complete text
            for each section of the report:
            {{
                "title": "Report title",
                "executive_summary": "Complete text for executive summary",
                "introduction": "Complete text for introduction",
                "methodology": "Complete text for methodology",
                "key_findings": "Complete text for key findings",
                "discussion": "Complete text for discussion",
                "research_gaps": "Complete text for research gaps",
                "future_directions": "Complete text for future directions",
                "conclusion": "Complete text for conclusion",
                "bibliography": ["citation1", "citation2", ...]
            }}
            """
            
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4000,
                system="You are a research report assistant that formats research findings into professional reports.",
                messages=[{"role": "user", "content": formatting_prompt}]
            )
            
            # Extract the content from the response
            response_content = response.content[0].text
            
            # Extract the JSON from the response
            report_json = self._extract_json(response_content)
            
            # Add metadata to the final output
            final_output = {
                "title": report_json.get("title", synthesis.title),
                "executive_summary": report_json.get("executive_summary", synthesis.summary),
                "introduction": report_json.get("introduction", f"Introduction to {synthesis.title}"),
                "methodology": report_json.get("methodology", "Methodology section"),
                "key_findings": report_json.get("key_findings", "Key findings section"),
                "discussion": report_json.get("discussion", "Discussion section"),
                "research_gaps": report_json.get("research_gaps", "Research gaps section"),
                "future_directions": report_json.get("future_directions", "Future directions section"),
                "conclusion": report_json.get("conclusion", "Conclusion section"),
                "bibliography": report_json.get("bibliography", synthesis.bibliography),
                "metadata": {
                    "research_question": synthesis.title,
                    "document_count": len(synthesis.document_coverage),
                    "finding_count": len(synthesis.key_findings),
                    "generation_date": datetime.now().isoformat(),
                    "research_id": synthesis.id
                }
            }
            
            return final_output
        except Exception as e:
            print(f"Error formatting research output: {str(e)}")
            # Return a basic formatted output with the synthesis data
            return {
                "title": synthesis.title,
                "executive_summary": synthesis.summary,
                "introduction": f"Introduction to {synthesis.title}",
                "methodology": "This research was conducted through a systematic review of literature.",
                "key_findings": "\n".join([f"- {finding.content}" for finding in synthesis.key_findings]),
                "discussion": "Discussion of the findings and their implications.",
                "research_gaps": "\n".join([f"- {gap}" for gap in synthesis.gaps_identified]),
                "future_directions": "\n".join([f"- {direction}" for direction in synthesis.future_directions]),
                "conclusion": f"In conclusion, this research on {synthesis.title} provides valuable insights.",
                "bibliography": synthesis.bibliography,
                "metadata": {
                    "research_question": synthesis.title,
                    "document_count": len(synthesis.document_coverage),
                    "finding_count": len(synthesis.key_findings),
                    "generation_date": datetime.now().isoformat(),
                    "research_id": synthesis.id
                }
            }
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from Claude's response text."""
        try:
            # Try to find JSON between triple backticks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If no JSON in backticks, try to find JSON between curly braces
            json_match = re.search(r'({[\s\S]*})', text)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
                
            # If still no JSON, try to parse the entire response
            return json.loads(text)
        except Exception as e:
            print(f"Error extracting JSON: {str(e)}")
            # Fallback: return a simple dictionary
            return {
                "error": "Failed to parse JSON",
                "raw_text": text[:500] + "..." if len(text) > 500 else text
            }
    
    def _format_citation(self, document: ResearchDocument) -> str:
        """Format a document as a citation."""
        try:
            authors = ", ".join(document.authors)
            return f"{authors}. ({document.publication_date}). {document.title}. {document.source}."
        except Exception as e:
            print(f"Error formatting citation: {str(e)}")
            return f"Citation for {document.title}"


def main():
    """Run the research agent."""
    try:
        # Get API key from environment variable or use a default for testing
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY environment variable not set.")
            print("Using placeholder value for demonstration purposes.")
            api_key = "your-api-key-here"  # This won't work for actual API calls
        
        # Initialize the research agent
        research_agent = DeepResearchAgent(
            api_key=api_key,
            search_api_key=None  # Using mock data instead
        )
        
        # Conduct research on a topic
        print("Starting research process...")
        research_output = research_agent.conduct_research(
            query="What are the latest advances in quantum computing and their potential impact on AI?",
            depth="standard"
        )
        
        # Print the research output
        print("\n" + "="*50)
        print(f"Research Report: {research_output['title']}")
        print("="*50)
        print("\nExecutive Summary:")
        print(research_output['executive_summary'])
        print("\nKey Findings:")
        print(research_output['key_findings'][:500] + "..." if len(research_output['key_findings']) > 500 else research_output['key_findings'])
        print("\nConclusion:")
        print(research_output['conclusion'])
        print("\n" + "="*50)
        
        # Save the result to a file
        output_file = "quantum_computing_research.json"
        with open(output_file, "w") as f:
            json.dump(research_output, f, indent=2)
        
        print(f"\nFull research report saved to {output_file}")
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")


if __name__ == "__main__":
    import asyncio
    try:
        # For Python 3.7+
        asyncio.run(main())
    except AttributeError:
        # Fallback for older Python versions
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except TypeError:
        # If main is not async (after removing async/await)
        main()