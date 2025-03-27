import streamlit as st
import fitz  # PyMuPDF
import os
import openai
from github import Github, GithubException # Import GithubException for more specific error handling if needed
from linkedin_api import Linkedin
import markdown_it
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
import csv
from datetime import datetime
from PIL import Image



# Function to check secrets (kept for debugging if needed, but call is commented out in main)
def check_secrets():
    """Verify that all required secrets are present (for debugging)."""
    try:
        st.write("Secrets available (Debug Info):")
        st.write(f"OPENAI_API_KEY: {'✓' if 'OPENAI_API_KEY' in st.secrets else '✗'}")
        st.write(f"GITHUB_TOKEN: {'✓' if 'general' in st.secrets and 'GITHUB_TOKEN' in st.secrets.general else '✗'}")
        st.write(f"LINKEDIN_USERNAME: {'✓' if 'general' in st.secrets and 'LINKEDIN_USERNAME' in st.secrets.general else '✗'}")
        st.write(f"LINKEDIN_PASSWORD: {'✓' if 'general' in st.secrets and 'LINKEDIN_PASSWORD' in st.secrets.general else '✗'}")
    except Exception as e:
        st.error(f"Error accessing secrets: {e}")

# Function to check file existence (kept for debugging, call commented out in main)
def check_file_existence():
    """Check if required files exist (for debugging)."""
    files_to_check = [
        "Binay_Resume.pdf",
        "profile_photo.JPG",
    ]
    st.write("File Existence Check (Debug Info):")
    missing_files = False
    for file in files_to_check:
        exists = os.path.exists(file)
        st.write(f"{file}: {'✓' if exists else '✗'}")
        if not exists:
            st.warning(f"File {file} is missing. This may cause issues.")
            missing_files = True
    # Create CSVs if they don't exist
    for csv_file in ["chat_log.csv", "feedback_log.csv"]:
        if not os.path.exists(csv_file):
            try:
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Optionally write headers
                    if csv_file == "chat_log.csv":
                        writer.writerow(["Timestamp", "User Message", "AI Response"])
                    elif csv_file == "feedback_log.csv":
                        writer.writerow(["Timestamp", "Feedback"])
                st.write(f"Created missing file: {csv_file}")
            except Exception as e:
                st.error(f"Could not create {csv_file}: {e}")
                missing_files = True

    if not missing_files:
         st.write("All checked files exist or were created.")


# Set Environment Variables and API Keys
try:
    # Ensure OpenAI key is set for Langchain/OpenAI components
    if "OPENAI_API_KEY" not in os.environ and "OPENAI_API_KEY" in st.secrets:
         os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    elif "OPENAI_API_KEY" not in os.environ:
        st.error("OpenAI API Key not found in environment variables or Streamlit secrets.")

    # Get other secrets
    GITHUB_TOKEN = st.secrets.get("general", {}).get("GITHUB_TOKEN")
    LINKEDIN_USERNAME = st.secrets.get("general", {}).get("LINKEDIN_USERNAME")
    LINKEDIN_PASSWORD = st.secrets.get("general", {}).get("LINKEDIN_PASSWORD")

    # Validate critical secrets needed for data fetching
    if not GITHUB_TOKEN:
        st.warning("GitHub Token not found in secrets ([general].GITHUB_TOKEN). GitHub data will be unavailable.")
    if not LINKEDIN_USERNAME or not LINKEDIN_PASSWORD:
        st.warning("LinkedIn credentials not found in secrets ([general].LINKEDIN_USERNAME/PASSWORD). LinkedIn data will be unavailable.")

except Exception as e:
    st.error(f"Error accessing secrets or setting environment variables: {e}")
    GITHUB_TOKEN = None
    LINKEDIN_USERNAME = None
    LINKEDIN_PASSWORD = None

# --- Resume Data ---
resume_data = {
    "Contact": {
        "Name": "Binay Siddharth",
        "Email": "binay.siddharth@gmail.com",
        "Phone": "+61 451 943 584",
        "LinkedIn": "https://www.linkedin.com/in/binaysiddharth",
        "GitHub": "https://github.com/binzidd",
        "Location": "Sydney, AU"
    },
    "Overview": {
        "Summary": """
        Results-driven Senior Manager with 7 years of progressive experience at Commonwealth Bank of Australia, specializing in transforming financial services through data. A proven leader in building high-performing teams, I excel at turning complex data into actionable insights and strategic solutions. From developing executive-level reporting hubs to pioneering Generative AI applications, I'm passionate about leveraging cutting-edge technology to optimize business performance and drive innovation. My expertise spans the full data lifecycle, from data engineering and modeling to impactful BI and advanced analytics. I thrive in collaborative environments, working closely with stakeholders to deliver solutions that make a tangible difference.
        """,
        "FeaturedWork": [
            {
                "name": "Uber Analytics",
                "url": "https://public.tableau.com/app/profile/binay5660/viz/MyUberAnalytics/MyUberJourney",
                "description": "Advanced Analytics Dashboard"
            },
            {
                "name": "Airbnb Tableau Community",
                "url": "https://public.tableau.com/app/profile/binay5660/viz/HeyAirbnbWhataremychoicesinSydney/AirbnbDashboard",
                "description": "Tableau Community Project"
            }
        ]
    },
    "Experience": [
        {
            "Title": "Senior Manager, Group BI Reporting (Insights and Data)",
            "Company": "Commonwealth Bank of Australia",
            "Dates": "2022 - Current",
            "Location": "Sydney, AU",
            "Description": """
            Collaborating with finance leaders on strategic data-driven decisions. Led high-performing BI Reporting Hub.
            Managed data assets for Executive Leadership, driving decision-making through interactive KPI reports.
            Championed BI innovation and upskilling. Leading GenAI solutions for business optimization.
            """,
            "Projects": [
                {
                    "Name": "Reporting hub for Executive Leadership, strategic measures | Data Lake and Reporting 📊",
                    "Description": "Uplifting manual processes with auto-ingest feeds and Tableau. 70% time saved and modern BI tools used for executive engagement."
                },
                {
                    "Name": "Process Efficiency and Data Lineage for Capital Engine ⚙️",
                    "Description": "Reduced Capital Production Month End lifecycle from 4 WD to 1.5 WD. Achieved 85% faster process and increased Data Visibility."
                },
                {
                    "Name": "ELT Scorecard | Strategic Metrics | Data Lake and Visualization Layer ☁️",
                    "DescriptionPoints": [
                        "Created conceptual data lake infrastructure and Extract Load Transform architecture.",
                        "Developed Tableau dashboards with drill-through to financial P&L.",
                        "Integrated data with robust lineage tracking."
                    ]
                },
                {
                    "Name": "Generative AI & AWS-Powered Business Solutions (Proof of Concept & Experimentation) 🤖",
                    "Dates": "2024-10 - Present", # Note: Check if this date is correct, seems like future
                    "DescriptionPoints": [
                        "**Intelligent Receipt Processing with AWS Textract:** Led PoC to production using Textract and Lambda. Explored GenAI (RAG, vector stores, intent-based routing).",
                        "**Natural Language to SQL with AWS and LangChain:** System for accountants to query databases using natural language, leveraging NLP, LangChain, and GenAI.",
                        "**Master Agentic Bot for Intelligent Task Routing:** Designed a 'master bot' using classification to understand user intent and direct requests.",
                        "**Key Technologies:** Python, LangChain, AWS (Textract, Lambda, S3, RDS, etc.), GenAI, RAG, Vector Databases."
                    ]
                }
            ]
        },
        {
            "Title": "Manager, BI Reporting",
            "Company": "IB&M Finance", # Consider adding (Subsidiary of CBA) if relevant for context
            "Dates": "2020 - 2022",
            "Location": "Sydney, AU",
            "Description": "Managed BI Reporting Projects, delivering data based solutions for financial insights.",
            "Projects": [
                {
                    "Name": "Project Spur | Data Transformation and Reporting 🔄",
                    "DescriptionPoints": [
                        "Ingested and modelled data from various source systems (Risk, Finance, Treasury, Capital).",
                        "Separated visualization/data layers, enabling dynamic refreshes and eliminating manual processes.",
                        "Achieved process efficiency."
                    ]
                }
            ]
        },
        {
            "Title": "Senior Analyst, Capital Tech",
            "Company": "GCARD", # Consider spelling out 'Group Corporate Affairs & Reporting Division' or similar if needed for clarity
            "Dates": "2019 - 2020",
            "Location": "Sydney, AU",
            "Description": "Senior Analyst role focused on Capital Tech projects.",
            "Projects": [
                {
                    "Name": "Daily Capital Reporting",
                    "Description": "Developed/maintained daily capital reporting, ensuring timely/accurate metrics."
                }
            ]
        },
        {
            "Title": "Analyst, Gems Operations",
            "Company": "GCARD", # Same as above regarding clarity
            "Dates": "2018 - 2019",
            "Location": "Sydney, AU",
            "Description": "Analyst role in Gems Operations.",
            "Projects": [
                {
                    "Name": "Loan IQ Data Feed Ingestion and Integration with Data Infrastructure 🔗",
                    "DescriptionPoints": [
                        "Captured document-based files and integrated with capital engine for RWA calculation.",
                        "Data Transformation/Processing in Alteryx with reconciliation.",
                        "Delivered reporting layer on monthly portfolio movement."
                    ]
                }
            ]
        },
       {
            "Title": "Analyst, University of Sydney Roles",
            "Company": "The University of Sydney",
            "Dates": "2017 - 2018",
            "Location": "Sydney, AU",
            "Description": "Held multiple roles focused on BI, analytics, and business analysis.",
            "Roles": [
                {
                    "SubTitle": "Analyst, BI and Analytics (Design and Reporting)",
                    "SubDates": "2017 - 2018",
                    "SubDescription": """Implemented agile analytics using Tableau and Alteryx. Data mining, cleansing, and transformation. Collaborated with product owners.""",
                     "Projects": [
                        {
                            "Name": "Canvas Reporting and Student Engagement dashboards 📈",
                            "DescriptionPoints": [
                                "Designed dashboards for Unit of Study co-ordinators to check engagement.",
                                "Extracted data from S3, APIs, data modeling, presentation layer.",
                                "Implemented RLS."
                            ]
                        },
                        {
                            "Name": "HDR Students Predictive model for Course Completion 🎯",
                            "DescriptionPoints": [
                                "Reporting on predictive model.",
                                "Call to action dashboards for DVC Grants."
                            ]
                        },
                        {
                            "Name": "Student Performance Forecasting and Classification 🤖",
                            "DescriptionPoints": [
                                "Developed forecasting models for student performance using ARIMA in Alteryx.",
                                "Implemented classification models using Random Forest for predicting student outcomes.",
                                "Utilized machine learning techniques to enhance predictive accuracy."
                            ]
                        }
                    ]
                },
                {
                    "SubTitle": "BI Dev, HDR Reporting",
                    "SubDates": "2018",
                    "SubDescription": "BI Development for HDR Reporting.",
                    "SubCompany": "IAP" # Consider spelling out 'Information and Analytics Portfolio' or similar
                },
                {
                    "SubTitle": "Business Analyst, Canvas",
                    "SubDates": "2017",
                    "SubDescription": "Business Analyst role for Canvas project.",
                    "SubCompany" : "IAP" # Same as above
                },
            ],
        },
       {
            "Title": "Tutoring Post Grads",
            "Company": "The University of Sydney",
            "Dates": "2019 - 2021",
            "Location": "Sydney, AU",
            "Description": "Tutoring and mentoring post-graduate students on BI tools and data analytics techniques within the Knowledge Management Systems course.",
           "Subject": "Knowledge Management Systems",
                    "SubjectDescription": """
                    This course provides a comprehensive introduction to Knowledge Management (KM) from both technological and organizational perspectives. It covers a range of KM-related topics through published papers, case studies, and other publications. Key areas include: KM Conceptual Foundations; Taxonomies of organizational knowledge and KM mechanisms; Case/Field Studies of KM Initiatives; Data Warehousing and OLAP/Business Analytics; Data, text, and web mining; Social media, crowdsourcing, and KM; Big data and actionable knowledge. The course includes detailed coverage of Business Intelligence Systems, with hands-on work using the BI (Online Analytical Processing - OLAP) tool, COGNOS.
                    """

        },
        {
            "Title": "Data Engineering Intern and Related Roles",
            "Company": "BizCubed Pty Ltd, Yahoo7, Internal Services", # Combined companies for title clarity
            "Dates": "2016 - 2018",
            "Location": "Sydney, AU",
            "Description": "Various roles in data engineering, development, and performance monitoring across different engagements.",
            "Roles": [
                {
                    "SubTitle": "Data Engineering Intern, Data and Platform Enablement",
                    "SubDates": "2016 - 2017",
                    "SubDescription": """Developed performance dashboards and pipelines. Designed "heartbeat" services and dimensional models. Liaised between teams.""",
                    "SubCompany": "BizCubed Pty Ltd",
                     "Projects" : [
                        {
                            "Name": "National Sales Report | Yahoo7 📰",
                            "DescriptionPoints": [
                                "Created summary reports on social media interaction.",
                                "Developed KPI and OKR dashboards.",
                                "Ingested streaming and transactional data."
                            ]
                        },
                        {
                            "Name": "Performance Monitoring Report | Managed Services 📊",
                            "DescriptionPoints": [
                                "Generated daily reports on server health.",
                                "Provided commentary on downtime/resolution.",
                                "Created One Pagers for Leadership."
                            ]
                        }
                    ]
                },
                {
                    "SubTitle": "Developer, Chanel Data",
                    "SubDates": "2018",
                    "SubDescription": "Developer role for Chanel Data project.",
                    "SubCompany": "Yahoo7", # Role performed while engaged via BizCubed or directly? Clarify if needed.
                },
                {
                    "SubTitle": "Junior Developer, Managed Services",
                    "SubDates": "2018",
                    "SubDescription": "Junior Developer role within Managed Services.",
                    "SubCompany": "BizCubed Pty Ltd (Internal Services)", # Clarified company context
                },
            ],
        },
        {
            "Title": "Production Support Analyst and Related Roles",
            "Company": "Adobe Systems Inc.",
            "Dates": "2012 - 2015",
            "Location": "Mumbai, India",
            "Description": "Multiple roles evolving from Analyst to Production Support and Business Analyst functions.",
            "Roles": [
                {
                    "SubTitle": "Production Support Analyst, Subscription Services",
                    "SubDates": "2012 - 2015", # Dates cover the whole Adobe period? Check if this specific role was shorter.
                    "SubDescription": """Managed cross-functional teams and scrum meetings. Metrics generation, reporting, and risk analysis.""",
                    "Projects": [
                        {
                            "Name": "Cush (Community Unified Social Hub) 🧑‍🤝‍🧑",
                            "DescriptionPoints": [
                                "Served as Technical Business Analyst for Adobe's internal social media platform.",
                                "Engaged with stakeholders, captured features, and drove development.",
                                "Provided break-fix coding support."
                            ]
                        },
                        {
                            "Name": "RSM (Retail Subscription Management) 💰",
                            "DescriptionPoints": [
                                "Automated SQL processes for data ingestion related to subscription management.",
                                "Provided Root Cause Analysis (RCA) for data discrepancies.",
                                "Created revenue-impacting reports for stakeholders."
                            ]
                        },
                        {
                            "Name": "User Analytics and Journey Optimization",
                            "DescriptionPoints": [
                                "Conducted user analytics to understand behavior and identify pain points within Adobe platforms.",
                                "Developed strategies and recommendations to optimize user journeys.",
                                "Utilized Adobe Experience Manager (AEM) for content and user experience analysis.",
                                "Collaborated with UI/UX and development teams to implement improvements for engagement/conversion."
                            ]
                        }
                    ]
                },
                {
                    "SubTitle": "Business Analyst, CUSH",
                    "SubDates": "2014 - 2015",
                    "SubDescription": "Dedicated Business Analyst role for the CUSH project.",
                    "SubCompany": "Digital Engineering" # Assumed team/division
                },
                {
                    "SubTitle": "Analyst, Web Content",
                    "SubDates": "2013 - 2014",
                    "SubDescription": "Analyst role focused on Web Content management and analysis.",
                    "SubCompany": "Digital Engineering" # Assumed team/division
                },
                {
                    "SubTitle": "Analyst, Subscription",
                    "SubDates": "2012 - 2013",
                    "SubDescription": "Initial Analyst role focused on Subscription services and data.",
                    "SubCompany": "RSM" # Assumed team/division
                },
            ],
        },
    ],
    "Certifications": [
        "Tableau: Community Leader, Data Scientist, Data Steward, Executive Sponsor, Desktop Specialist", # Consider listing individually if preferred
        "Alteryx: Core & Advanced Certified",
        "AWS: Analytics Service Overview, Certified Cloud Practitioner",
        "Snowflake Hands-on Essentials: Data Warehouse, Data Applications",
        "Tableau Certified Designer", # Check if this is distinct from Desktop Specialist or other Tableau certs
        "Alteryx 2020 Certified Specialist" # Check if this is superseded by Core/Advanced
    ],
    "References": "Available on Request", # Standard placeholder
    "Areas_of_Improvement": [
        "Strategic Messaging - Enhancing the narrative around my strategic contributions.",
        "Technology (Continuous Learning) - Staying ahead of the curve in rapidly evolving technologies.",
        "Public Speaking - Refining presentation skills for larger audiences.",
        "Cross-functional Collaboration - Deepening collaboration across diverse business units."
        ]
}

# --- Skill Scores Data ---

skill_scores = {
    "Leadership & Strategy": {
        "Stakeholder Management": (4.5, "Extensive experience collaborating with finance leaders and executive teams.",
                                  [
                                      {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": None},
                                      {"job": "Production Support Analyst", "company": "Adobe Systems Inc.", "project": "Cush (Community Unified Social Hub)"} # Added example link
                                  ]),
        "Strategic Thinking": (5, "Demonstrated in developing strategic initiatives and leading complex projects.",
                              [
                                  {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "ELT Scorecard"},
                                  {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "Generative AI & AWS-Powered Business Solutions"}
                              ]),
        "Team Building": (5, "Led and mentored high-performing teams across multiple organizations.",
                         [
                             {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "BI Reporting Hub"},
                             {"job": "Manager, BI Reporting", "company": "IB&M Finance", "project": None},
                             {"job": "Production Support Analyst", "company": "Adobe Systems Inc.", "project": None} # General team interaction
                         ]),
        "Vision Alignment": (4.5, "Experience aligning project goals with broader organizational objectives.",
                            [
                                {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "Reporting hub for Executive Leadership"},
                                {"job": "Analyst, BI and Analytics (Design and Reporting)", "company": "The University of Sydney", "project": "Canvas Reporting and Student Engagement dashboards"} # Updated job title
                            ]),
        "Risk Management": (4, "Involved in risk analysis and mitigation strategies during product launches.",
                           [
                               {"job": "Production Support Analyst", "company": "Adobe Systems Inc.", "project": "RSM (Retail Subscription Management)"} # Specific project link
                           ])
    },
    "Generative AI & Emerging Tech": {
        "AWS Services (Textract, Lambda)": (4, "Hands-on experience with AWS Textract and Lambda for GenAI solutions.",
                                            [
                                                {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "Generative AI & AWS-Powered Business Solutions - Receipt Processing"}
                                            ]),
        "NLP & SQL": (4, "Developed NLP-driven solutions for natural language to SQL conversion.",
                      [
                          {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "Generative AI & AWS-Powered Business Solutions - NLP to SQL"}
                      ]),
        "RAG & Vector Stores": (4, "Exploration and implementation of RAG and vector DBs for enhanced GenAI.",
                               [
                                   {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "Generative AI & AWS-Powered Business Solutions - RAG & Vector Stores"}
                               ]),
        "LangChain": (4, "Utilized LangChain framework for building GenAI applications.",
                      [
                          {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "Generative AI & AWS-Powered Business Solutions - LangChain Applications"}
                      ]),
        "Agentic AI": (4, "Designed and implemented agentic AI systems for intelligent task routing.",
                       [
                           {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "Generative AI & AWS-Powered Business Solutions - Master Agentic Bot"}
                       ])
    },
    "Data Engineering": {
        "Data Pipelines": (5, "Expertise in building robust and scalable data pipelines.",
                          [
                              {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "Reporting hub for Executive Leadership"},
                              {"job": "Data Engineering Intern", "company": "BizCubed Pty Ltd", "project": "National Sales Report | Yahoo7"} # Matches updated experience
                          ]),
        "Data Modelling": (5, "Extensive experience in data modeling (dimensional and relational).",
                          [
                              {"job": "Manager, BI Reporting", "company": "IB&M Finance", "project": "Project Spur"},
                              {"job": "Data Engineering Intern", "company": "BizCubed Pty Ltd", "project": "Performance Monitoring Report | Managed Services"} # Matches updated experience
                          ]),
        "SQL": (5, "Highly proficient in SQL for data manipulation, analysis, and automation.",
                [
                    {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "Process Efficiency and Data Lineage for Capital Engine"},
                    {"job": "Production Support Analyst", "company": "Adobe Systems Inc.", "project": "RSM (Retail Subscription Management)"} # Matches updated experience
                ]),
        "Python": (4, "Proficient in Python for data engineering and application development.",
                   [
                       {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "Generative AI & AWS-Powered Business Solutions"},
                       {"job": "Analyst, BI and Analytics (Design and Reporting)", "company": "The University of Sydney", "project": "Canvas Reporting and Student Engagement dashboards"} # Matches updated experience
                   ]),
        "Cloud (AWS / Snowflake)": (4, "Experience with AWS and Snowflake for data warehousing / analytics.",
                                 [
                                     {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "Generative AI & AWS-Powered Solutions"},
                                     {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "ELT Scorecard"}
                                 ]),
        "APIs": (4, "Utilized APIs for data integration and extraction.",
                 [
                     {"job": "Analyst, BI and Analytics (Design and Reporting)", "company": "The University of Sydney", "project": "Canvas Reporting and Student Engagement dashboards"}, # Matches updated experience
                     {"job": "Analyst, Gems Operations", "company": "GCARD", "project": "Loan IQ Data Feed Ingestion"} # Matches updated experience
                 ])
    },
    "BI & Analytics": {
        "BI Product Dev": (5, "Led development of BI products from concept to delivery.",
                           [
                               {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "Reporting hub for Executive Leadership"},
                               {"job": "Manager, BI Reporting", "company": "IB&M Finance", "project": "Project Spur"},
                               {"job": "Analyst, BI and Analytics (Design and Reporting)", "company": "The University of Sydney", "project": "Canvas Reporting and Student Engagement dashboards"} # Matches updated experience
                           ]),
        "Data Storytelling": (5, "Exceptional ability to communicate insights through narratives / visualizations.",
                             [
                                 {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "ELT Scorecard"},
                                 {"job": None, "company": None, "project": "Uber Analytics Dashboard"}, # Public projects remain
                                 {"job": None, "company": None, "project": "Airbnb Tableau Community Project"} # Public projects remain
                             ]),
        "Tableau": (5, "Extensive expertise in Tableau for interactive dashboards.",
                    [
                        {"job": "Senior Manager, Group BI Reporting", "company": "Commonwealth Bank of Australia", "project": "Reporting hub for Executive Leadership"},
                        {"job": "Manager, BI Reporting", "company": "IB&M Finance", "project": "Project Spur"},
                        {"job": "Analyst, BI and Analytics (Design and Reporting)", "company": "The University of Sydney", "project": "Canvas Reporting and Student Engagement dashboards"} # Matches updated experience
                    ]),
        "Alteryx / Power BI": (4, "Proficient in Alteryx for data prep and Power BI for reporting.", # Added Power BI based on data
                           [
                               {"job": "Analyst, Gems Operations", "company": "GCARD", "project": "Loan IQ Data Feed Ingestion"}, # Matches updated experience
                               {"job": "Analyst, BI and Analytics (Design and Reporting)", "company": "The University of Sydney", "project": "Canvas Reporting and Student Engagement dashboards"} # Matches updated experience
                           ]),
         "Machine Learning (Alteryx/Python)": (4, "Applied ML techniques for forecasting and classification.", # Added ML skill
                           [
                               {"job": "Analyst, BI and Analytics (Design and Reporting)", "company": "The University of Sydney", "project": "Student Performance Forecasting and Classification"}
                           ]),
    },
    "Process & Delivery": {
        "Agile / Iterative": (4, "Experienced in Agile and iterative development methodologies.",
                           [
                               {"job": "Analyst, BI and Analytics (Design and Reporting)", "company": "The University of Sydney", "project": "Canvas Reporting and Student Engagement dashboards"}, # Matches updated experience
                               {"job": "Production Support Analyst", "company": "Adobe Systems Inc.", "project": "Cush (Community Unified Social Hub)"} # Matches updated experience
                           ]),
        "Human-Centered Design": (5, "Focus on user-centered design principles for effective solutions.",
                               [
                                   {"job": "Production Support Analyst", "company": "Adobe Systems Inc.", "project": "User Analytics and Journey Optimization"}, # Matches updated experience
                                   {"job": "Analyst, BI and Analytics (Design and Reporting)", "company": "The University of Sydney", "project": "Canvas Reporting and Student Engagement dashboards"} # Matches updated experience
                               ])
    }
}

# --- Helper Functions with Caching ---

@st.cache_data
def cached_extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file with caching."""
    return extract_text_from_pdf(pdf_path)

@st.cache_data
def cached_extract_github_info(_github_token): 
    if not GITHUB_TOKEN:
         st.warning("GitHub Token not available, skipping GitHub data extraction.")
         return None
    return extract_github_info(GITHUB_TOKEN)

@st.cache_data
def cached_extract_linkedin_info(_username, _password): # Use _ for cache invalidation triggers
    """Extracts relevant information from LinkedIn with caching."""
    if not LINKEDIN_USERNAME or not LINKEDIN_PASSWORD:
        st.warning("LinkedIn credentials not available, skipping LinkedIn data extraction.")
        return None
    # Pass actual credentials here
    return extract_linkedin_info(LINKEDIN_USERNAME, LINKEDIN_PASSWORD)

@st.cache_resource # Caches the actual vectorstore object
def cached_create_vectorstore(resume_data, resume_text, github_data_json, linkedin_data_json): # Pass data as JSON serializable for caching if needed, or rely on object hash
    """Creates and populates the Chroma vector store with caching."""

    return create_vectorstore(resume_data, resume_text, github_data_json, linkedin_data_json)

@st.cache_resource # Caches the RAG chain object
def cached_create_rag_chain(_vectorstore): # Use _vectorstore to depend on the vectorstore instance
    """Creates the RAG chain with caching."""
    return create_rag_chain(_vectorstore)


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except FileNotFoundError:
        st.error(f"❌ PDF file not found at path: '{pdf_path}'. Please ensure it's in the correct location.")
        return None
    except Exception as e:
        st.error(f"❌ Error processing PDF '{pdf_path}': {e}")
        return None
    return text

def extract_github_info(github_token):

    if not github_token:
        st.warning("GitHub token is missing. Cannot fetch GitHub data.")
        return None
    try:
        g = Github(github_token)
        user = g.get_user() # Gets the authenticated user
        github_data = {
            "profile": {
                "username": user.login,
                "bio": user.bio or "N/A",
                "name": user.name or "N/A",
                "company": user.company or "N/A",
                "location": user.location or "N/A",
                "email": user.email or "N/A",
                "url": user.html_url,
                "followers": user.followers,
                "following": user.following,
                "public_repos": user.public_repos,
            },
            "repositories": [],
        }

        # Fetch repositories - handles pagination automatically
        repos = user.get_repos(sort="updated", direction="desc") # Get most recently updated first
        repo_count = 0
        for repo in repos:
            repo_data = {
                "name": repo.name,
                "description": repo.description or "No description provided.",
                "url": repo.html_url,
                "language": repo.language or "N/A",
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "last_updated": repo.updated_at.strftime("%Y-%m-%d"),
                "readme": "README not found or could not be decoded.", # Default value
            }
            try:
                # Try fetching README (common names)
                readme_content = None
                for readme_name in ["README.md", "README.rst", "README"]:
                    try:
                        readme_content = repo.get_contents(readme_name)
                        break # Found one
                    except GithubException as ge:
                        if ge.status == 404:
                            continue # Try next name
                        else:
                            raise # Re-raise other GitHub errors
                if readme_content:
                    decoded_content = readme_content.decoded_content.decode('utf-8', errors='ignore') # Decode safely
                    # Basic Markdown rendering (consider more robust library if needed)
                    # md = markdown_it.MarkdownIt()
                    # repo_data["readme"] = md.render(decoded_content)
                    repo_data["readme"] = decoded_content # Store raw text for vectorization

            except Exception as readme_e:
                 # Log this error for debugging, but don't stop the process
                 print(f"Warning: Could not process README for repo '{repo.name}': {readme_e}")
                 pass # Keep default message

            github_data["repositories"].append(repo_data)
            repo_count += 1

        print(f"Successfully processed {repo_count} GitHub repositories.") # Debugging print to console
        return github_data

    except GithubException as ge:
         if ge.status == 401:
             st.error("❌ GitHub Error: Authentication failed. Check your GITHUB_TOKEN permissions (needs 'repo' and 'user:read' scopes) and validity.")
         elif ge.status == 403:
             st.error("❌ GitHub Error: Forbidden. Rate limit exceeded or insufficient permissions. Wait and try again later.")
         else:
            st.error(f"❌ GitHub Error: {ge}. Status: {ge.status}, Data: {ge.data}")
         return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while accessing GitHub: {e}")
        return None


def extract_linkedin_info(username, password):
    """Extracts relevant information from LinkedIn."""
    if not username or not password:
        st.warning("LinkedIn username or password missing. Cannot fetch LinkedIn data.")
        return None
    try:
        api = Linkedin(username, password)
        profile_id = "binaysiddharth" # Make sure this is correct
        profile = api.get_profile(profile_id)

        if not profile:
            st.error(f"❌ LinkedIn profile not found for ID: '{profile_id}'. Check the ID or login credentials.")
            return None

        linkedin_data = {
            "profile": {
                "summary": profile.get("summary", "N/A"),
                "headline": profile.get("headline", "N/A"),
                "location": profile.get("locationName", "N/A"), # Example: adding location
                "geo_country": profile.get("geoCountryName", "N/A"),
                "industry": profile.get("industryName", "N/A")
            },
            "experience": [],
            "education": [],
            "skills": [skill.get("name", "Unknown Skill") for skill in profile.get("skills", [])], # Safer access
        }

        # Process Experience
        if 'experience' in profile and isinstance(profile['experience'], list):
            for exp in profile["experience"]:
                start_date = exp.get("timePeriod", {}).get("startDate", {})
                end_date = exp.get("timePeriod", {}).get("endDate", {})
                start_year = start_date.get("year", "") if start_date else ""
                start_month = start_date.get("month", "") if start_date else "" # Optional: Add month
                end_year = end_date.get("year", "") if end_date else "Present" # Handle ongoing roles
                end_month = end_date.get("month", "") if end_date else ""

                dates_str = f"{start_month}/{start_year}" if start_month else str(start_year)
                dates_str += f" - {end_month}/{end_year}" if end_month and end_year != "Present" else f" - {end_year}"

                linkedin_data["experience"].append(
                    {
                        "title": exp.get("title", "N/A"),
                        "company": exp.get("companyName", "N/A"),
                        "location": exp.get("locationName", "N/A"), # Add location if available
                        "dates": dates_str,
                        "description": exp.get("description", "N/A"),
                    }
                )
        else:
            st.warning("No LinkedIn experience data found or data is in an unexpected format.")

        # Process Education
        if 'education' in profile and isinstance(profile['education'], list):
            for edu in profile['education']:
                start_date = edu.get("timePeriod", {}).get("startDate", {})
                end_date = edu.get("timePeriod", {}).get("endDate", {})
                start_year = start_date.get("year", "") if start_date else ""
                end_year = end_date.get("year", "") if end_date else "N/A" # Or "Present" if applicable

                dates_str = f"{start_year} - {end_year}" if start_year or end_year else "N/A"

                linkedin_data['education'].append({
                    'schoolName': edu.get('schoolName', 'N/A'),
                    'degreeName': edu.get('degreeName', 'N/A'),
                    'fieldOfStudy': edu.get('fieldOfStudy', 'N/A'),
                    "dates": dates_str,
                    'description': edu.get('description', 'N/A') # Add description if available
                })
        else:
             st.warning("No LinkedIn education data found or data is in an unexpected format.")

        print("Successfully processed LinkedIn data.") # Debug print to console
        return linkedin_data

    except Exception as e:
        # Catch more specific exceptions if possible (e.g., authentication errors)
        st.error(f"❌ Error accessing LinkedIn: {e}. Check credentials and profile ID ('{profile_id}').")
        # You might be logged out or require 2FA. The linkedin_api library can be fragile.
        return None


def create_vectorstore(resume_text, resume_data, github_data, linkedin_data):
    """Creates and populates the Chroma vector store."""
    all_documents = []
    text_splitter_large = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    text_splitter_medium = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    # Convert resume_data dict to string if it's not already text
    if resume_data and isinstance(resume_data, dict):
        resume_data_str = "\n".join([f"{key}: {value}" for key, value in resume_data.items()])
        resume_data_chunks = text_splitter_medium.split_text(resume_data_str)
        for i, chunk in enumerate(resume_data_chunks):
            all_documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": "resume", "chunk": i},
                )
            )

    # Add resume documents (text version)
    if resume_text and isinstance(resume_text, str):
        resume_chunks = text_splitter_medium.split_text(resume_text)
        for i, chunk in enumerate(resume_chunks):
            all_documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": "resume", "chunk": i},
                )
            )

    # Rest of the function remains the same...
    # Add GitHub documents
    if github_data:
        # Profile info
        profile_content = f"GitHub Profile ({github_data['profile']['username']}): Name: {github_data['profile']['name']}, Bio: {github_data['profile']['bio']}, Location: {github_data['profile']['location']}, Company: {github_data['profile']['company']}. Followers: {github_data['profile']['followers']}, Public Repos: {github_data['profile']['public_repos']}."
        all_documents.append(
            Document(
                page_content=profile_content,
                metadata={"source": "github", "type": "profile", "username": github_data['profile']['username']},
            )
        )
        # Repositories
        for repo in github_data.get("repositories", []):
            repo_summary = f"GitHub Repo: {repo['name']}\nLanguage: {repo['language']}\nStars: {repo['stars']}\nDescription: {repo['description']}\nLast Updated: {repo['last_updated']}"
            all_documents.append(
                Document(
                    page_content=repo_summary,
                    metadata={"source": "github", "type": "repo_summary", "repo_name": repo['name'], "url": repo['url']},
                )
            )
            if repo["readme"] and repo["readme"] != "README not found or could not be decoded.":
                readme_chunks = text_splitter_large.split_text(repo["readme"])
                for i, chunk in enumerate(readme_chunks):
                    all_documents.append(
                        Document(
                            page_content=f"GitHub README ({repo['name']}): {chunk}",
                            metadata={"source": "github", "type": "readme", "repo_name": repo['name'], "chunk": i, "url": repo['url']},
                        )
                    )

    # Add LinkedIn documents
    if linkedin_data:
        # Profile summary
        profile_content = f"LinkedIn Profile Summary: {linkedin_data['profile']['headline']}. {linkedin_data['profile']['summary']}. Location: {linkedin_data['profile']['location']}, Industry: {linkedin_data['profile']['industry']}."
        all_documents.append(
            Document(
                page_content=profile_content,
                metadata={"source": "linkedin", "type": "profile"},
            )
        )
        # Experience
        for i, exp in enumerate(linkedin_data.get("experience", [])):
            exp_content = f"LinkedIn Experience {i+1}: {exp['title']} at {exp['company']} ({exp['dates']}, Location: {exp.get('location', 'N/A')}).\nDescription: {exp['description']}"
            all_documents.append(
                Document(
                    page_content=exp_content,
                    metadata={"source": "linkedin", "type": "experience", "company": exp['company'], "title": exp['title']},
                )
            )
         # Education
        for i, edu in enumerate(linkedin_data.get("education", [])):
            edu_content = f"LinkedIn Education {i+1}: {edu['degreeName']} in {edu['fieldOfStudy']} at {edu['schoolName']} ({edu['dates']}).\nDescription: {edu.get('description', 'N/A')}"
            all_documents.append(
                Document(
                    page_content=edu_content,
                    metadata={"source": "linkedin", "type": "education", "school": edu['schoolName'], "degree": edu['degreeName']},
                )
            )

        # Skills (as a single document)
        if linkedin_data.get("skills"):
            all_documents.append(
                Document(
                    page_content=f"LinkedIn Skills: {', '.join(linkedin_data['skills'])}",
                    metadata={"source": "linkedin", "type": "skills"},
                )
            )

    if not all_documents:
        st.error("❌ No documents were generated from the sources. Cannot create vector store.")
        return None

    # Create and populate the vector store
    embeddings = OpenAIEmbeddings()
    persist_directory = "chroma_db_interactive_resume"

    try:
        vectorstore = Chroma.from_documents(
            documents=all_documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"Vectorstore created/updated in '{persist_directory}' with {len(all_documents)} documents.")
        return vectorstore
    except Exception as e:
        st.error(f"❌ Failed to create or update vector store: {e}")
        return None
# --- RAG Chain ---
def create_rag_chain(vectorstore):
    """Creates the RAG chain using the provided vector store."""
    if not vectorstore:
        st.error("❌ Cannot create RAG chain: Vectorstore is not available.")
        return None

    try:
        retriever = vectorstore.as_retriever(search_kwargs={'k': 5}) 

        llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0.1, max_tokens=500) 
        prompt_template = """You are Binay Siddharth's professional AI assistant. Your purpose is to answer questions about Binay's skills, experience, projects, and career based *only* on the information provided in the context below.

        Context Sources: Resume, GitHub profile/repositories, LinkedIn profile.

        Instructions:
        1.  Answer the user's question concisely and professionally.
        2.  Base your answer *strictly* on the provided context.
        3.  If the context does not contain the information needed to answer the question, clearly state that the information is not available in the provided documents (resume, GitHub, LinkedIn). Do not make assumptions or use external knowledge.
        4.  If relevant, mention the source of the information (e.g., "According to the resume...", "On GitHub, the project...").
        5.  Format your answers clearly. Use bullet points for lists if appropriate.
        6.  Do not mention the context itself in the final answer unless specifying the source.

        Context:
        {context}

        Question:
        {input}

        Answer:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)

        rag_chain = (
            {"context": retriever, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG chain created successfully.")
        return rag_chain

    except Exception as e:
        st.error(f"❌ Failed to create RAG chain: {e}")
        return None

# --- Streamlit App Functions (UI Rendering) ---

def configure_page():
    """Configures the Streamlit page settings."""
    st.set_page_config(
        page_title=f"{resume_data.get('Contact', {}).get('Name', 'Interactive')} Resume",
        page_icon="📄",
        layout="wide"
    )

def create_sidebar():
    """Creates the sidebar content including contact information and navigation."""
    st.sidebar.header(f"{resume_data.get('Contact', {}).get('Name', 'Name Not Found')}")

    # Profile Picture
    try:
        profile_image = Image.open("profile_photo.JPG")
        st.sidebar.image(profile_image, width=150, caption=resume_data.get('Contact', {}).get('Name'))
    except FileNotFoundError:
        st.sidebar.warning("Profile photo 'profile_photo.JPG' not found.")
    except Exception as e:
        st.sidebar.error(f"Error loading profile photo: {e}")

    # Contact Info - Cleaned
    contact = resume_data.get('Contact', {})
    st.sidebar.markdown(f"**Email:** {contact.get('Email', 'N/A')}")
    st.sidebar.markdown(f"**Phone:** {contact.get('Phone', 'N/A')}")
    st.sidebar.markdown(f"**Location:** {contact.get('Location', 'N/A')}")
    if contact.get('LinkedIn'):
        st.sidebar.markdown(f"**LinkedIn:** [{contact.get('LinkedIn').replace('https://','')}]({contact.get('LinkedIn')})") # Display cleaner link text
    if contact.get('GitHub'):
        st.sidebar.markdown(f"**GitHub:** [{contact.get('GitHub').replace('https://','')}]({contact.get('GitHub')})") # Display cleaner link text

    st.sidebar.markdown("---") # Divider

    # Navigation
    sections = ["Ask Binay", "Overview & Skills", "Experience & Projects", "Feedback", "Download PDF"]
    section_icons = {"Overview & Skills": "🚀", "Experience & Projects": "💼", "Ask Binay": "🤖", "Feedback": "📝", "Download PDF": "⬇️"} # Define icons here

    if 'selected_section' not in st.session_state:
        st.session_state['selected_section'] = "Ask Binay" # Default section

    def update_selected_section(section):
        st.session_state['selected_section'] = section

    for section_name in sections:
        # Using use_container_width makes buttons fill sidebar width
        st.sidebar.button(
            f"{section_icons.get(section_name, '')} {section_name}",
            key=f"nav_button_{section_name}",
            on_click=update_selected_section,
            args=(section_name,),
            use_container_width=True,
            type="secondary" if st.session_state['selected_section'] != section_name else "primary" # Highlight selected
        )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"© {datetime.now().year} {resume_data.get('Contact', {}).get('Name', '')}")


def render_overview_and_skills_section():
    """Renders the 'Overview & Skills' section."""
    st.title(f"Overview & Skills 🚀") # Use icon directly

    st.header("Executive Summary")
    st.markdown(resume_data.get('Overview', {}).get('Summary', 'Summary not available.').strip()) # Safer access

    st.subheader("Featured Work")
    featured_work = resume_data.get('Overview', {}).get('FeaturedWork', [])
    if featured_work:
        for work in featured_work:
            st.markdown(f"- **[{work.get('name', 'N/A')}]({work.get('url', '#')})**: {work.get('description', '')}")
    else:
        st.markdown("_No featured work listed._")

    st.subheader("Skills Proficiency Showcase")
    st.caption("Click on a skill category, then a skill to see related experiences.") # Use caption for subtle text

    if 'selected_skill_category' not in st.session_state:
        st.session_state['selected_skill_category'] = next(iter(skill_scores), None) # Default to first category
    if 'selected_skill' not in st.session_state:
        st.session_state['selected_skill'] = None

    skill_category_col, blank_1, skill_detail_col, blank_2, skill_experience_col = st.columns([0.15,0.1, 0.25,0.1, 0.35])

    with skill_category_col:
        st.markdown("##### Skill Categories") # Smaller heading
        skill_categories_list = list(skill_scores.keys())
        for category_name in skill_categories_list:
            # Highlight selected category
            button_type = "primary" if st.session_state['selected_skill_category'] == category_name else "secondary"
            if st.button(category_name, key=f"category_button_{category_name}", use_container_width=True, type=button_type):
                st.session_state['selected_skill_category'] = category_name
                st.session_state['selected_skill'] = None # Reset skill selection when category changes

    with skill_detail_col:
        selected_category = st.session_state['selected_skill_category']
        if selected_category:
            st.markdown(f"##### Skills in {selected_category}")
            if selected_category in skill_scores:
                for skill_name, skill_info in skill_scores[selected_category].items():
                    score, tooltip_text, _ = skill_info[:3] # Unpack first 3 elements safely
                    skill_button_key = f"skill_button_{selected_category}_{skill_name}" # Ensure unique key
                    button_type = "primary" if st.session_state['selected_skill'] == skill_name else "secondary"

                    if st.button(f"{skill_name}", key=skill_button_key, use_container_width=True, type=button_type):
                        st.session_state['selected_skill'] = skill_name

                    # Display score and tooltip below the button
                    full_icons = "🟢" * int(score)
                    half_icon = "🟡" if score % 1 >= 0.5 else ""
                    empty_icons = "⚪" * (5 - int(score) - (1 if half_icon else 0))
                    st.markdown(f"Proficiency: {full_icons}{half_icon}{empty_icons} ({score}/5)")
                    st.caption(tooltip_text) # Use caption for tooltip text
                    st.markdown("---") # Divider between skills
            else:
                st.write("No skills listed for this category.")
        else:
            st.write("Select a category on the left.")

    with skill_experience_col:
        selected_skill = st.session_state['selected_skill']
        st.markdown("##### Related Experience Highlights")
        if selected_skill and st.session_state['selected_skill_category']:
            st.markdown(f"**Skill:** {selected_skill}")
            selected_category_skills = skill_scores.get(st.session_state['selected_skill_category'], {})
            skill_data = selected_category_skills.get(selected_skill)

            if skill_data and len(skill_data) > 2 and skill_data[2]: # Check if experience links exist
                experience_links = skill_data[2]
                for link in experience_links:
                    job_title = link.get("job", "Various Roles")
                    company = link.get("company", "Various Companies")
                    project_name = link.get("project", None) # Use None to check existence easily

                    st.markdown(f"- **{job_title}**, *{company}*")
                    if project_name:
                        st.markdown(f"  - Project: *{project_name}*")
                    st.caption("---") # Small divider
            else:
                st.markdown("_No specific experience highlights linked for this skill._")
        else:
            st.markdown("_Click on a skill in the middle column to view related experience._")

    # Moved Certifications and Areas of Improvement to avoid clutter in columns
    st.markdown("---") # Main divider
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Certifications")
        certifications = resume_data.get('Certifications', [])
        if certifications:
            for cert in certifications:
                st.markdown(f"- ✨ {cert}")
        else:
            st.markdown("_No certifications listed._")

    with col2:
        st.subheader("Areas of Improvement / Future Focus") # Renamed for positivity
        areas = resume_data.get("Areas_of_Improvement", [])
        if areas:
            for area in areas:
                st.markdown(f"- {area}")
        else:
             st.markdown("_No specific areas listed._")


def render_experience_and_projects_section():
    """Renders the 'Experience & Projects' section."""
    st.title(f"Experience & Projects 💼")

    all_years = set()
    # Safely iterate through experience, handling potential missing keys
    for exp in resume_data.get("Experience", []):
        dates_str = exp.get("Dates", "")
        # Basic year extraction (can be made more robust)
        years_in_date = [int(y.strip()) for y in dates_str.replace("Current", str(datetime.now().year)).replace("Present", str(datetime.now().year)).split('-') if y.strip().isdigit()]
        all_years.update(years_in_date)

        # Also check project dates within experience
        if "Projects" in exp:
            for project in exp.get("Projects", []):
                proj_dates_str = project.get("Dates", "")
                proj_years = [int(y.strip()) for y in proj_dates_str.replace("Present", str(datetime.now().year)).split('-') if y.strip().isdigit()]
                all_years.update(proj_years)
        # Also check project dates within Roles (nested structure)
        if "Roles" in exp:
            for role in exp.get("Roles", []):
                 if "Projects" in role:
                     for project in role.get("Projects", []):
                        proj_dates_str = project.get("Dates", "")
                        proj_years = [int(y.strip()) for y in proj_dates_str.replace("Present", str(datetime.now().year)).split('-') if y.strip().isdigit()]
                        all_years.update(proj_years)


    if not all_years:
        min_year, max_year = 2010, datetime.now().year # Default range if no years found
    else:
        min_year = min(all_years)
        max_year = max(all_years)

    # Ensure min_year is not greater than max_year
    if min_year > max_year:
        min_year = max_year

    # Date Range Slider
    start_year, end_year = st.slider(
        "Filter Experience by Year Range:",
        min_value=2012,
        max_value=max_year,
        value=(2012, max_year), # Default to full range
        help="Adjust the slider to filter work experience and projects within the selected years."
    )

    st.markdown("---") # Divider

    # Filter and Display Experience
    filtered_count = 0
    for job in resume_data.get('Experience', []):
        dates_str = job.get("Dates", "")
        job_years = [int(y.strip()) for y in dates_str.replace("Current", str(datetime.now().year)).replace("Present", str(datetime.now().year)).split('-') if y.strip().isdigit()]

        job_start_year = min(job_years) if job_years else 0
        job_end_year = max(job_years) if job_years else 9999

        # Check if the job's period overlaps with the selected slider range
        if job_end_year >= start_year and job_start_year <= end_year:
            filtered_count += 1
            with st.container():
                st.subheader(f"**{job.get('Title', 'N/A')}**")
                st.markdown(f"*{job.get('Company', 'N/A')}* ({job.get('Dates', 'N/A')}) - {job.get('Location', 'N/A')}")

                # Use expander for details
                with st.expander("Details & Projects", expanded=False): # Default to collapsed
                    # Handle nested Roles structure first
                    if "Roles" in job and job["Roles"]:
                        for role in job['Roles']:
                            st.markdown(f"**{role.get('SubTitle', 'Role')}** ({role.get('SubDates', 'N/A')})")
                            if role.get('SubCompany') and role['SubCompany'] != job.get('Company'):
                                st.markdown(f"*{role['SubCompany']}*") # Display sub-company if different
                            st.markdown(role.get("SubDescription", "").strip())

                            # Display projects within the role
                            role_projects = role.get('Projects', [])
                            if role_projects:
                                st.markdown("**Key Projects:**")
                                for project in role_projects:
                                     # Check project date filter
                                    display_project = True
                                    proj_dates_str = project.get("Dates", "")
                                    proj_years_list = [int(y.strip()) for y in proj_dates_str.replace("Present", str(datetime.now().year)).split('-') if y.strip().isdigit()]
                                    if proj_years_list:
                                        proj_start = min(proj_years_list)
                                        proj_end = max(proj_years_list)
                                        if not (proj_end >= start_year and proj_start <= end_year):
                                            display_project = False

                                    if display_project:
                                        project_name = project.get('Name', 'Unnamed Project')
                                        if project.get('Description'):
                                            st.markdown(f"- **{project_name}**: {project['Description']}")
                                        elif project.get('DescriptionPoints'):
                                            st.markdown(f"- **{project_name}**:")
                                            for point in project['DescriptionPoints']:
                                                st.markdown(f"  - {point.strip()}") # Use simple list marker
                                        else:
                                             st.markdown(f"- **{project_name}**")

                            if role.get("Subject"): # Handle Subject if present
                                 st.markdown(f"**Subject Focus: {role['Subject']}**")
                                 st.markdown(role.get("SubjectDescription", "").strip())
                            st.markdown("---") # Divider between roles

                    # Handle top-level Description and Projects if no Roles
                    else:
                        st.markdown("**Role Summary:**")
                        st.markdown(job.get('Description', 'N/A').strip())

                        job_projects = job.get('Projects', [])
                        if job_projects:
                            st.markdown("**Key Projects:**")
                            for project in job_projects:
                                # Check project date filter
                                display_project = True
                                proj_dates_str = project.get("Dates", "")
                                proj_years_list = [int(y.strip()) for y in proj_dates_str.replace("Present", str(datetime.now().year)).split('-') if y.strip().isdigit()]
                                if proj_years_list:
                                    proj_start = min(proj_years_list)
                                    proj_end = max(proj_years_list)
                                    if not (proj_end >= start_year and proj_start <= end_year):
                                        display_project = False

                                if display_project:
                                    project_name = project.get('Name', 'Unnamed Project')
                                    if project.get('Description'):
                                        st.markdown(f"- **{project_name}**: {project['Description']}")
                                    elif project.get('DescriptionPoints'):
                                        st.markdown(f"- **{project_name}**:")
                                        for point in project['DescriptionPoints']:
                                            st.markdown(f"  - {point.strip()}")
                                    else:
                                        st.markdown(f"- **{project_name}**")

                st.markdown("---") # Divider between job experiences

    if filtered_count == 0:
        st.info(f"No work experience found within the selected date range ({start_year} - {end_year}).")


def render_ask_binay_section():
    """Renders the AI Chatbot section."""
    st.header(f"Ask Binay (AI Assistant) 🤖")
    st.markdown("Ask questions about my resume, skills, GitHub projects, and professional experience. The AI uses information from my uploaded resume, GitHub, and LinkedIn profiles.")
    st.caption("Example questions: 'Summarize Binay's experience with Generative AI.', 'What are some key projects listed on the resume?', 'Tell me about the 'Project Spur' mentioned in the experience.'")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi there! I'm Binay's AI assistant. Ask me anything about his professional background based on his resume, GitHub, and LinkedIn data."}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Your question..."):
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if RAG chain exists
        if 'rag_chain' not in st.session_state or st.session_state.rag_chain is None:
            st.error("Sorry, the AI assistant is not available at the moment. The RAG chain failed to initialize.")
            st.session_state.messages.append({"role": "assistant", "content": "Apologies, I couldn't process that as my underlying systems are not ready. Please check the application setup."})
            # Log error?
            return # Stop processing this input

        # Generate and display AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
                try:
                    # --- Using invoke (simpler for non-streaming) ---
                    response = st.session_state.rag_chain.invoke(prompt)
                    full_response = response
                    message_placeholder.markdown(full_response)

                    # --- OPTIONAL: Streaming Response ---
                    # response_stream = st.session_state.rag_chain.stream(prompt)
                    # for chunk in response_stream:
                    #     full_response += chunk
                    #     message_placeholder.markdown(full_response + "▌") # Add cursor effect
                    # message_placeholder.markdown(full_response) # Final response without cursor
                    # ------------------------------------

                except Exception as e:
                    error_message = f"Sorry, I encountered an error trying to answer that: {e}"
                    st.error(error_message)
                    full_response = error_message

        # Add AI response to history and log
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        log_chat(prompt, full_response)


def render_feedback_section():
    """Renders the 'Feedback' section."""
    st.header(f"Feedback 📝")
    st.markdown("Your feedback is valuable and helps improve this interactive resume. Please share your thoughts, suggestions, or report any issues!")

    with st.form("feedback_form"):
        feedback_name = st.text_input("Your Name (Optional)")
        feedback_email = st.text_input("Your Email (Optional - for follow-up)")
        feedback_rating = st.slider("Overall Impression (1=Poor, 5=Excellent)", 1, 5, 3)
        feedback_text = st.text_area("Your Feedback:", height=150, placeholder="Enter your feedback here...")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            if feedback_text:
                try:
                    log_feedback(feedback_name, feedback_email, feedback_rating, feedback_text)
                    st.success("Thank you for your feedback! It has been recorded.")
                except Exception as e:
                    st.error(f"Sorry, there was an error logging your feedback: {e}")
            else:
                st.warning("Please enter some feedback before submitting.")


def render_download_pdf_section():
    """Renders the 'Download PDF' section."""
    st.header(f"Download Resume PDF ⬇️")
    st.markdown("Click the button below to download the static PDF version of my resume.")

    pdf_path = "Binay_Resume.pdf" # Ensure this matches the filename exactly

    try:
        # Check if file exists before attempting to open
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as file:
                st.download_button(
                    label="📥 Download PDF Resume",
                    data=file,
                    file_name="Binay_Siddharth_Resume.pdf", # You can set a user-friendly download name
                    mime="application/pdf"
                )
        else:
             st.error(f"❌ Error: The resume PDF file ('{pdf_path}') was not found in the application directory.")

    except Exception as e:
        st.error(f"❌ An error occurred while preparing the PDF for download: {e}")


# --- Logging Functions ---

def log_chat(user_message, ai_response):
    """Logs chat messages to a CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = "chat_log.csv"
    try:
        # Create file and header if it doesn't exist
        if not os.path.exists(log_file):
             with open(log_file, "w", newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Timestamp", "User Message", "AI Response"])

        # Append the new log entry
        with open(log_file, "a", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, user_message, ai_response])
    except Exception as e:
        print(f"Error writing to chat log: {e}") # Log error to console


def log_feedback(name, email, rating, feedback_text):
    """Logs feedback messages to a CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = "feedback_log.csv"
    try:
         # Create file and header if it doesn't exist
        if not os.path.exists(log_file):
             with open(log_file, "w", newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Timestamp", "Name", "Email", "Rating", "Feedback"])

        # Append the new log entry
        with open(log_file, "a", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, name, email, rating, feedback_text])
    except Exception as e:
        print(f"Error writing to feedback log: {e}") # Log error to console


# --- Main Application Execution ---

def main():
    configure_page()
    create_sidebar()



    # --- RAG Setup (Run once per session) ---
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None # Initialize as None
        with st.spinner("Initializing AI Assistant... (Loading data, this might take a moment)"):
            # 1. Extract Data (Use cached functions)
            pdf_path = "Binay_Resume.pdf"
            resume_text = cached_extract_text_from_pdf(pdf_path)
            # Pass tokens/credentials to cached functions for potential cache invalidation
            github_data = cached_extract_github_info(GITHUB_TOKEN)
            linkedin_data = cached_extract_linkedin_info(LINKEDIN_USERNAME, LINKEDIN_PASSWORD)

            # Check if essential data loaded (Resume is critical)
            if not resume_text:
                st.error("❌ Critical Error: Failed to load resume text. AI Assistant cannot be initialized.")
                return # Stop execution if resume fails

            # Check for non-critical data loading issues (display warnings but continue)
            if github_data is None:
                 st.warning("⚠️ Could not load GitHub data. AI assistant's knowledge will be limited.")
            if linkedin_data is None:
                 st.warning("⚠️ Could not load LinkedIn data. AI assistant's knowledge will be limited.")

            # 2. Create Vectorstore (Use cached function)
            # Pass actual data here. Caching is handled by @st.cache_resource
            vectorstore = cached_create_vectorstore(resume_data,resume_text, github_data, linkedin_data)

            if vectorstore is None:
                st.error("❌ Critical Error: Failed to create vector store. AI Assistant cannot be initialized.")
                return # Stop execution

            # 3. Create RAG Chain (Use cached function)
            # Pass the created vectorstore instance
            st.session_state.rag_chain = cached_create_rag_chain(vectorstore)

            if st.session_state.rag_chain:
                st.success("✅ AI Assistant is ready!")
            else:
                # Error message already shown by create_rag_chain or vectorstore creation
                st.error("❌ AI Assistant initialization failed. Please check logs or configuration.")

    # --- Render Selected Section ---
    selected_section = st.session_state.get('selected_section', "Ask Binay") # Default safely

    if selected_section == "Overview & Skills":
        render_overview_and_skills_section()
    elif selected_section == "Experience & Projects":
        render_experience_and_projects_section()
    elif selected_section == "Ask Binay":
        render_ask_binay_section()
    elif selected_section == "Feedback":
        render_feedback_section()
    elif selected_section == "Download PDF":
        render_download_pdf_section()
    else:
        # Fallback to default section if state is somehow invalid
        st.session_state['selected_section'] = "Ask Binay"
        render_ask_binay_section()


if __name__ == "__main__":
    main()