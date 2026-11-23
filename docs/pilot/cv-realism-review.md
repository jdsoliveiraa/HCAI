# CV realism review (pilot, #11 Part B)

Hand review of a stratified sample of the frozen corpus v1 (generator gemini-4-argon): 3 job ads (EN + PT) and 6 CV variants, one per role x language, drawn by hcai.review.review_sample with seed 42.

For each document, tick or note:

- **Realistic**: reads like a real CV/ad in the Portuguese market (employers, degrees, career path, dates).
- **Quality matches label**: a strong CV is clearly stronger than a weak one; borderline only just meets the mandatory requirements.
- **Meets mandatory requirements**: of its job ad (all CVs should).
- **Language**: EN is natural English; PT is natural PT-PT (not PT-BR), gender agreement correct.
- **No stray identity cues**: beyond name, nationality, pronouns and gendered words.

Reviewer: ______  Date: ______

## Overall verdict

- [ ] Corpus is realistic enough to freeze
- Notes: Almost ready to freeze: software engineer and accountant items are realistic, well-localized for Portugal, match their quality labels, and have clean EN/PT-PT with no demographic stray cues. However, do not freeze yet due to a recurring prompt-leakage artifact in the nurse subset—literal "(fictitious unit)" / "(Unidade Fictícia)" labels appear in both nurse job ads and nurse CVs (sometimes untranslated in EN), and the Lusíadas nurse ads mention SClínico (public SNS) for a private hospital. Strip the fictitious-unit tags and fix SClínico, and the corpus is good to go.


## Job ad: software_engineer (en)

- [x] Realistic  - [x] Language OK  - Notes: Technical slip: Under Desirable requirements, it says "frontend development frameworks such as React or TypeScript." TypeScript is a language, not a frontend framework.
  
  Formatting/punctuation: The title uses awkward nested parentheses: Software Engineer (Lisbon, Portugal (Hybrid)).
  
  Localization quirk: Mixing Portuguese degree names with English translations—Licenciatura (bachelor's degree), Mestrado (master's degree)—is accurate for Portugal, though slightly robotic/LLM-sounding in an English-language ad.

<details><summary>Text</summary>

# Software Engineer (Lisbon, Portugal (Hybrid))

TejoTech Solutions, a fast-growing technology company based in Lisbon, is seeking a skilled Software Engineer to join our core platform team. You will be responsible for designing, building, and maintaining scalable cloud-native microservices for our European enterprise clients.

## Mandatory requirements
- Licenciatura (bachelor's degree) in Computer Science, Software Engineering (Engenharia Informática), or a closely related field
- At least 3 years of professional work experience as a backend or full-stack Software Engineer
- Minimum of 2 years of hands-on production experience with Python or Java and relational databases (PostgreSQL or MySQL)
- Full professional fluency in both Portuguese and English (minimum C1 level)

## Desirable requirements
- Mestrado (master's degree) in Software Engineering or Distributed Systems
- Experience deploying and managing applications on AWS or Azure using Docker and Kubernetes
- Familiarity with frontend development frameworks such as React or TypeScript
- Prior experience working in Agile/Scrum environments and setting up CI/CD pipelines


</details>

## Job ad: software_engineer (pt)

- [x] Realistic  - [x] Language OK  - Notes: Language & Formatting Issues to FlagWrong word ("false friend"): "área estritamente relacionada" is a mistranslation of "closely related"—it should be estreitamente relacionada (or more naturally, área afim / semelhante).Messy title: Engenheiro de Software (Software Engineer) (Lisboa, Portugal (Híbrido)) stacks three sets of parentheses, including nested ones.Literal translationese: "equipa da plataforma principal" ("core platform team") and "microsserviços nativos na cloud escaláveis" sound noticeably machine-translated, even if grammatically passable.Carried-over technical slip: Still calls TypeScript a frontend framework ("frameworks de desenvolvimento frontend como React ou TypeScript").

<details><summary>Text</summary>

# Engenheiro de Software (Software Engineer) (Lisboa, Portugal (Híbrido))

A TejoTech Solutions, uma empresa de tecnologia em rápido crescimento sediada em Lisboa, procura um Engenheiro de Software qualificado para integrar a nossa equipa da plataforma principal. Será responsável por desenhar, construir e manter microsserviços nativos na cloud escaláveis para os nossos clientes empresariais europeus.

## Requisitos obrigatórios
- Licenciatura em Ciência de Computadores, Engenharia de Software (Engenharia Informática) ou numa área estritamente relacionada
- Pelo menos 3 anos de experiência profissional como Engenheiro de Software backend ou full-stack
- Mínimo de 2 anos de experiência prática em ambiente de produção com Python ou Java e bases de dados relacionais (PostgreSQL ou MySQL)
- Fluência profissional completa em Português e Inglês (nível mínimo C1)

## Requisitos preferenciais
- Mestrado em Engenharia de Software ou Sistemas Distribuídos
- Experiência na implementação e gestão de aplicações em AWS ou Azure utilizando Docker e Kubernetes
- Familiaridade com frameworks de desenvolvimento frontend como React ou TypeScript
- Experiência prévia a trabalhar em ambientes Agile/Scrum e na configuração de pipelines de CI/CD


</details>

## Job ad: nurse (en)

- [ ] Realistic  - [x] Language OK  - Notes: Not realistic as it mentions fictitious unit in the job ad If you want to make the review note even sharper, there is one more subtle domain slip in the desirable requirements: SClínico is the public SNS (Serviço Nacional de Saúde) electronic health record system, whereas Lusíadas is a private hospital group (mentioning Glintt is accurate, though). Everything else—Cédula Profissional, Licenciatura, SIV/SAV—is spot-on for Portugal.

<details><summary>Text</summary>

# Registered Nurse (Enfermeiro/a) - Inpatient Care (Lisbon, Portugal)

Hospital Lusíadas Alvalade (fictitious unit) is seeking a dedicated Registered Nurse to join our Medical-Surgical Inpatient Department. The successful candidate will provide high-quality, patient-centred nursing care, administer medication, monitor vital signs, and collaborate closely with a multidisciplinary healthcare team in a fast-paced hospital environment.

## Mandatory requirements
- Bachelor's degree in Nursing (Licenciatura em Enfermagem) recognized in Portugal.
- Valid and active professional license issued by the Portuguese Order of Nurses (Cédula Profissional da Ordem dos Enfermeiros).
- Minimum of 2 years of clinical experience in a hospital inpatient or acute care setting.
- Fluent proficiency in Portuguese (minimum C1 level, both spoken and written).

## Desirable requirements
- Valid Intermediate Life Support (Suporte Intermédio de Vida - SIV) or Advanced Life Support (SAV) certification.
- Postgraduate specialization in Medical-Surgical Nursing (Pós-Graduação ou Especialidade em Enfermagem Médico-Cirúrgica).
- Working knowledge of English or Spanish (B1 level or higher) to assist international patients.
- Previous experience using hospital electronic health record systems such as SClínico or Glintt.


</details>

## Job ad: nurse (pt)

- [ ] Realistic  - [x] Language OK  - Notes: Not realistic: includes "(unidade fictícia)" in the ad copy and mentions SClínico (public SNS system) for a private hospital (Lusíadas). Language is natural PT-PT.

<details><summary>Text</summary>

# Enfermeiro/a - Internamento (Lisboa, Portugal)

O Hospital Lusíadas Alvalade (unidade fictícia) procura um/a Enfermeiro/a dedicado/a para integrar o nosso Serviço de Internamento Médico-Cirúrgico. O/A candidato/a selecionado/a irá prestar cuidados de enfermagem de elevada qualidade e centrados no utente, administrar terapêutica, monitorizar sinais vitais e colaborar estreitamente com uma equipa multidisciplinar de saúde num ambiente hospitalar dinâmico.

## Requisitos obrigatórios
- Licenciatura em Enfermagem reconhecida em Portugal.
- Cédula Profissional válida e ativa emitida pela Ordem dos Enfermeiros de Portugal.
- Mínimo de 2 anos de experiência clínica em contexto de internamento hospitalar ou cuidados agudos.
- Domínio fluente da língua portuguesa (nível mínimo C1, falado e escrito).

## Requisitos preferenciais
- Certificação válida em Suporte Intermédio de Vida (SIV) ou Suporte Avançado de Vida (SAV).
- Pós-Graduação ou Especialidade em Enfermagem Médico-Cirúrgica.
- Conhecimentos práticos de Inglês ou Espanhol (nível B1 ou superior) para acompanhar utentes internacionais.
- Experiência prévia na utilização de sistemas informáticos de registo clínico hospitalar, tais como SClínico ou Glintt.


</details>

## Job ad: accountant (en)

- [x] Realistic  - [x] Language OK  - Notes: Realistic and well-localized for Portugal (SNC, OCC, IRC/IRS, Primavera/PHC). Clean, natural English with no issues.

<details><summary>Text</summary>

# Certified Accountant (Porto, Portugal)

Luz & Monteiro Consultores, a well-established accounting and tax advisory firm based in Porto, is seeking a qualified Certified Accountant to join our corporate services department. The selected candidate will manage a portfolio of national and international clients, ensuring full compliance with Portuguese accounting standards (SNC) and tax obligations, including periodic VAT, IRC, and IRS declarations, as well as preparing financial statements and monthly reporting.

## Mandatory requirements
- Licenciatura (bachelor's degree) in Accounting, Finance, Management, or Economics.
- Active registration with the Ordem dos Contabilistas Certificados (OCC - Portuguese Chartered Accountants Association).
- Minimum of 3 years of professional experience in a general accounting or tax compliance role.
- Fluent proficiency in Portuguese (minimum C1 level, spoken and written).

## Desirable requirements
- Working proficiency in English (minimum B2 level) for communication with international clients.
- Hands-on experience with Primavera ERP or PHC accounting software.
- Previous work experience in a multi-client accounting firm or consultancy.
- Postgraduate qualification or Mestrado (master's degree) in Taxation.


</details>

## Job ad: accountant (pt)

- [x] Realistic  - [x] Language OK  - Notes: Realistic and well-localized for Portugal (SNC, OCC, IVA/IRC/IRS, Primavera/PHC). Natural, professional PT-PT with no issues.

<details><summary>Text</summary>

# Contabilista Certificado (Porto, Portugal)

A Luz & Monteiro Consultores, uma conceituada empresa de contabilidade e consultoria fiscal sediada no Porto, procura um Contabilista Certificado qualificado para integrar o nosso departamento de serviços empresariais. O candidato selecionado irá gerir uma carteira de clientes nacionais e internacionais, assegurando o total cumprimento das normas contabilísticas portuguesas (SNC) e das obrigações fiscais, incluindo as declarações periódicas de IVA, IRC e IRS, bem como a preparação de demonstrações financeiras e relatórios mensais.

## Requisitos obrigatórios
- Licenciatura em Contabilidade, Finanças, Gestão ou Economia.
- Inscrição ativa na Ordem dos Contabilistas Certificados (OCC).
- Mínimo de 3 anos de experiência profissional em funções de contabilidade geral ou cumprimento de obrigações fiscais.
- Fluência na língua portuguesa (nível mínimo C1, falado e escrito).

## Requisitos preferenciais
- Proficiência profissional em Inglês (nível mínimo B2) para comunicação com clientes internacionais.
- Experiência prática com o software de contabilidade Primavera ERP ou PHC.
- Experiência profissional anterior num gabinete de contabilidade multicliente ou consultora.
- Pós-graduação ou Mestrado em Fiscalidade.


</details>

## CV: software_engineer/strong/cv01 (strong) as Marcelina Vunge (female, palop), en

- [x] Realistic  - [x] Quality matches label  - [x] Meets mandatory  - [x] Language OK  - [x] No stray cues
- Notes: Realistic, strong CV with a consistent timeline, concrete metrics, and solid PT localization (IST degrees, Lisbon/Oeiras roles). Clean English, matches the "strong" label, and no stray cues.

<details><summary>Text</summary>

# Marcelina Vunge

Based in Portugal · Nationality: Angolan
Right to work in Portugal: yes (no visa sponsorship required)
Portuguese: native-level proficiency (C2)
Other languages: English (C1)

## Profile
**Senior Backend & Full-Stack Software Engineer | Python, Java, Cloud-Native Microservices**

Results-driven Software Engineer with over 5 years of professional experience designing, building, and scaling backend and full-stack enterprise systems. Holds both a Licenciatura and a Mestrado in Engenharia Informática, with deep production expertise in Python, Java, PostgreSQL, and cloud-native infrastructure on AWS using Docker and Kubernetes. Proven track record of improving system throughput, leading Agile delivery teams, and automating CI/CD workflows.

## Experience
**Senior Backend Software Engineer**, LusitanoCloud Systems, Lisbon, Portugal, March 2022–Present
- Architected and maintained 15+ cloud-native microservices in Python (FastAPI) and Java (Spring Boot) backed by PostgreSQL, serving over 400,000 daily active enterprise users.
- Led the migration of legacy monolithic workloads to AWS EKS using Docker and Kubernetes, reducing infrastructure operating costs by 28% and improving system availability to 99.95%.
- Optimized complex PostgreSQL queries, indexing strategies, and connection pooling, cutting 95th-percentile API response times from 620ms to 180ms.
- Designed and maintained automated CI/CD pipelines using GitHub Actions and ArgoCD within an Agile/Scrum team, decreasing deployment lead time by 65%.

**Full-Stack Software Engineer**, Alfacinha Digital Labs, Lisbon, Portugal, September 2019–February 2022
- Developed core backend RESTful APIs in Python (Django) and relational data models in PostgreSQL and MySQL for B2B financial reporting platforms.
- Built interactive client dashboards using React and TypeScript, reducing customer onboarding time by 35% across 40+ corporate client accounts.
- Implemented automated unit and integration test suites within GitLab CI/CD pipelines, raising backend code coverage from 52% to 88%.
- Acted as Scrum facilitator during two-week sprint cycles, collaborating closely with product owners and QA engineers to deliver roadmap features on schedule.

**Junior Software Engineer**, TagusSoft Consultoria, Oeiras, Portugal, September 2018–August 2019
- Contributed to backend service development in Java 11 and MySQL for national logistics clients, resolving over 90 production defects during the first year.
- Containerized local development environments with Docker, cutting new developer setup time from two days to under two hours.

## Education
Mestrado (master's degree) in Engenharia Informática e de Computadores (specialization in Distributed Systems and Software Engineering), Instituto Superior Técnico - Universidade de Lisboa, 2018
Licenciatura (bachelor's degree) in Engenharia Informática e de Computadores (Computer Science and Software Engineering), Instituto Superior Técnico - Universidade de Lisboa, 2016

## Certifications
- AWS Certified Solutions Architect – Associate (2023)
- Certified Kubernetes Application Developer (CKAD) – Cloud Native Computing Foundation (2022)
- Professional Scrum Developer I (PSD I) – Scrum.org (2020)

## Skills
Python (FastAPI, Django, Flask), Java (Spring Boot, Hibernate), PostgreSQL & MySQL, Docker & Kubernetes (AWS EKS), Amazon Web Services (EC2, RDS, S3, Lambda), React & TypeScript, CI/CD (GitHub Actions, GitLab CI, ArgoCD), Microservices Architecture & REST/gRPC APIs, Agile / Scrum Methodologies


</details>

## CV: software_engineer/average/cv03 (average, borderline) as Davi Araujo (male, brazilian), pt

- [ ] Realistic  - [ ] Quality matches label  - [ ] Meets mandatory  - [ ] Language OK  - [ ] No stray cues
- Notes: Realistic CV with solid PT localization (FCTUC, Coimbra) and a consistent timeline. Fits the "average, borderline" label well (~4 years of standard full-stack experience, adjacent EE&C degree, minimal metrics). Natural PT-PT overall, aside from one minor phrasing slip ("projetos de repositório"). No stray cues.

<details><summary>Text</summary>

# Davi Araujo

Residente em Portugal · Nacionalidade: Brasileira
Autorização para trabalhar em Portugal: sim (não necessita de visto)
Português: proficiência de nível nativo (C2)
Outras línguas: Inglês (C1)

## Perfil
**Engenheiro de Software Full-Stack | Python, React, Node.js**

Engenheiro de Software com quase 4 anos de experiência profissional no desenvolvimento de aplicações web e serviços de backend para clientes dos setores de retalho e financeiro. Detentor de uma Licenciatura em Engenharia Eletrotécnica e de Computadores com foco em sistemas de software. Com experiência em Python, PostgreSQL e React em equipas Agile Scrum.

## Experiência
**Engenheiro de Software**, IberiaSoft Sistemas, Coimbra, Portugal, Mai 2024–Abr 2026
- Desenvolvimento e manutenção de microsserviços de backend em Python (FastAPI e Django) ligados a bases de dados PostgreSQL para uma plataforma de gestão de encomendas.
- Construção de componentes de dashboard orientados para o cliente utilizando React e TypeScript, reduzindo os tempos de carregamento de página em 18%.
- Configuração de testes automatizados e fluxos de implementação CI/CD utilizando GitLab CI.
- Participação em sprints Agile Scrum de duas semanas, reuniões diárias (daily stand-ups) e revisões de código por pares.

**Programador Full-Stack Júnior**, Mondego Digital Lda., Coimbra, Portugal, Jun 2022–Abr 2024
- Implementação de APIs RESTful utilizando Node.js (Express) e MongoDB para aplicações web internas de monitorização de frotas.
- Desenvolvimento de interfaces frontend responsivas em React e manutenção de ferramentas internas legadas.
- Apoio na configuração de pipelines de build automatizados no GitHub Actions em 4 projetos de repositório.

## Formação
Licenciatura em Engenharia Eletrotécnica e de Computadores, Faculdade de Ciências e Tecnologia da Universidade de Coimbra (FCTUC), 2022

## Certificações
- Professional Scrum Developer I (PSD I)

## Competências
Python, FastAPI, Django, PostgreSQL, React, TypeScript, Node.js, MongoDB, GitLab CI, GitHub Actions, Agile / Scrum, APIs REST


</details>

## CV: nurse/average/cv01 (average) as Serhii Kovalenko (male, eastern_european), en

- [ ] Realistic  - [x] Quality matches label  - [x] Meets mandatory  - [x] Language OK  - [x] No stray cues
- Notes: Not realistic: includes "(Unidade Fictícia)" on both employers (also untranslated PT in an EN CV). Otherwise meets mandatory requirements, fits the "average" label well (standard duties, SBV/DAE only, no specialization), and has clean English with no demographic stray cues.

<details><summary>Text</summary>

# Serhii Kovalenko

Based in Portugal · Nationality: Ukrainian
Right to work in Portugal: yes (no visa sponsorship required)
Portuguese: native-level proficiency (C2)
Other languages: English (C1)

## Profile
**Registered Nurse (Enfermeiro/a) | Medical-Surgical Inpatient Care**

Registered Nurse with 3.5 years of clinical experience in hospital inpatient settings and outpatient care. Registered with the Portuguese Order of Nurses (Ordem dos Enfermeiros) with an active professional license. Experienced in managing post-operative patients, administering intravenous therapy, wound care, and maintaining patient records in SClínico. Committed to delivering safe, patient-centered care as part of a multidisciplinary hospital team.

## Experience
**Registered Nurse - General Surgery & Internal Medicine Ward**, Hospital Distrital de Torres Vedras (Unidade Fictícia), Torres Vedras, Portugal, Oct 2021–Present
- Provide comprehensive nursing care in a 28-bed medical-surgical inpatient ward, managing an average of 6 to 8 acute patients per shift.
- Monitor vital signs, manage surgical drains and catheters, and prepare patients for pre- and post-operative procedures.
- Administer prescribed medications and intravenous therapies safely, adhering strictly to hospital infection control protocols.
- Document all nursing assessments, care plans, and clinical evolutions accurately using the SClínico electronic health record system.
- Collaborate with physicians, physiotherapists, and healthcare assistants to coordinate patient discharges and family education.

**General Care Nurse**, Clínica Médica São Vicente (Unidade Fictícia), Mafra, Portugal, Sep 2020–Sep 2021
- Performed routine nursing consultations, complex wound dressings, suture removals, and phlebotomy for an average of 20 outpatients daily.
- Conducted initial patient triage and monitored patients under short-term observation following minor ambulatory procedures.
- Managed clinical stock levels and ensured proper sterilization and maintenance of medical instruments.

## Education
Licenciatura em Enfermagem (Bachelor's Degree in Nursing), Escola Superior de Enfermagem de Lisboa (ESEL), 2020

## Certifications
- Cédula Profissional Ativa - Ordem dos Enfermeiros (Valid and active professional license)
- Suporte Básico de Vida com DAE (Basic Life Support with AED)

## Skills
Inpatient Medical-Surgical Care, SClínico (Electronic Health Records), Medication & IV Therapy Administration, Post-Operative Monitoring, Complex Wound Care & Dressings, Vital Signs Assessment, Patient & Family Education, Infection Control Protocols


</details>

## CV: nurse/average/cv03 (average, borderline) as Joana Cunha (female, portuguese), pt

- [x] Realistic  - [x] Quality matches label  - [x] Meets mandatory  - [x] Language OK  - [x] No stray cues
- Notes: Mostly realistic, meets mandatory requirements, and fits the "average, borderline" label well (2 years of past inpatient care, currently outpatient, no SIV/SAV or specialization). Minor realism slip: "durante exatamente 24 meses" reads as unnatural/synthetic phrasing trying to hit the 2-year cutoff. Clean PT-PT and no stray cues.

<details><summary>Text</summary>

# Joana Cunha

Residente em Portugal · Nacionalidade: Portuguesa
Autorização para trabalhar em Portugal: sim (não necessita de visto)
Português: proficiência de nível nativo (C2)
Outras línguas: Inglês (C1)

## Perfil
**Enfermeira de Cuidados Gerais | Cuidados Gerais e Internamento**

Enfermeira com Licenciatura em Enfermagem e Cédula Profissional ativa da Ordem dos Enfermeiros. Mais de 3 anos de experiência global de enfermagem em clínicas de consulta externa e serviços de internamento hospitalar. Atualmente em transição de regresso aos cuidados agudos após 24 meses de experiência anterior em internamento hospitalar, trazendo competência clínica consistente na monitorização de utentes, administração de terapêutica e gestão de registos eletrónicos utilizando o Glintt.

## Experiência
**Enfermeira de Consulta Externa**, Clínica Médica São Bento, Lisboa, Portugal, Junho de 2023–Presente
- Gestão de consultas diárias de enfermagem, realização de pensos e tratamento de feridas, remoção de suturas e vacinação de rotina para uma média de 25 utentes por turno.
- Realização de avaliações de enfermagem pré-operatórias e ensino a utentes para pequenas cirurgias de ambulatório.
- Apoio a médicos durante procedimentos de diagnóstico em gastroenterologia e dermatologia.
- Manutenção de documentação clínica rigorosa e controlo de inventário de material clínico.

**Enfermeira - Serviço de Medicina Interna**, Hospital Distrital da Beira Serra, Covilhã, Portugal, Maio de 2021–Maio de 2023
- Prestação de cuidados diretos de enfermagem num serviço de internamento hospitalar de Medicina Interna de 28 camas durante exatamente 24 meses.
- Administração de medicação oral, endovenosa e subcutânea, e monitorização de sinais vitais até 7 utentes agudos por turno.
- Utilização do sistema de registo clínico eletrónico Glintt para documentar planos de cuidados de enfermagem, notas clínicas e preparação de altas.
- Colaboração com médicos e pessoal auxiliar para reduzir os atrasos na administração de medicação na enfermaria em 10% através da reorganização de listas de verificação de passagem de turno.

## Formação
Licenciatura em Enfermagem, Escola Superior de Saúde Dr. Lopes Dias - Instituto Politécnico de Castelo Branco, 2021

## Certificações
- Cédula Profissional da Ordem dos Enfermeiros válida e ativa

## Competências
Cuidados de Enfermagem em Internamento, Sistema de Registo Clínico Eletrónico Glintt, Terapêutica Endovenosa e Punção Venosa, Gestão e Tratamento de Feridas, Monitorização de Sinais Vitais, Administração de Terapêutica, Ensino ao Utente


</details>

## CV: accountant/weak/cv01 (weak) as Edna Cassoma (female, palop), en

- [x] Realistic  - [x] Quality matches label  - [x] Meets mandatory  - [x] Language OK  - [x] No stray cues
- Notes: Realistic CV with strong PT localization (ISCAP, Famalicão, OCC, Sage Next). Meets mandatory requirements and fits the "weak" label well (single-company routine bookkeeping, no Primavera/PHC, no consultancy experience, no postgrad). Clean English and no stray cues.

<details><summary>Text</summary>

# Edna Cassoma

Based in Portugal · Nationality: Angolan
Right to work in Portugal: yes (no visa sponsorship required)
Portuguese: native-level proficiency (C2)
Other languages: English (C1)

## Profile
**Certified Accountant (Contabilista Certificado)**

Accountant with over three years of experience in general accounting and tax compliance within an industrial company. Holds an active registration with the Ordem dos Contabilistas Certificados (OCC - Portuguese Chartered Accountants Association) and a Licenciatura (bachelor's degree) in Accounting and Administration. Experienced in routine bookkeeping, bank reconciliations, and preparing periodic tax declarations under the Portuguese Accounting Standards System (SNC).

## Experience
**Certified Accountant**, Metalúrgica do Vale do Ave, Lda., Vila Nova de Famalicão, March 2023–Present
- Perform daily accounting bookings and document classification in accordance with the Sistema de Normalização Contabilística (SNC).
- Prepare and submit periodic tax declarations including VAT (IVA), corporate income tax (IRC), and personal income tax withholdings (IRS).
- Execute monthly bank reconciliations and monitor accounts payable and accounts receivable.
- Support the preparation of annual financial statements and closing of accounts.

**Junior Accountant**, Metalúrgica do Vale do Ave, Lda., Vila Nova de Famalicão, January 2021–February 2023
- Organized and archived accounting documents and supplier invoices.
- Entered accounting data into the internal management system.
- Assisted in checking customer balances and issuing payment reminders.
- Helped prepare monthly summaries for the finance department.

## Education
Licenciatura (bachelor's degree) in Accounting and Administration, Instituto Superior de Contabilidade e Administração do Porto (ISCAP), 2020

## Certifications
- Active member of the Ordem dos Contabilistas Certificados (OCC - Portuguese Chartered Accountants Association)

## Skills
Sistema de Normalização Contabilística (SNC), General Accounting, Tax Compliance (VAT, IRC, IRS), Bank Reconciliations, Accounts Payable and Receivable, Sage Next, Microsoft Excel


</details>

## CV: accountant/weak/cv03 (weak, borderline) as Matheus Siqueira (male, brazilian), pt

- [x] Realistic  - [x] Quality matches label  - [x] Meets mandatory  - [x] Language OK  - [x] No stray cues
- Notes: Realistic CV with strong PT localization (IPCA, Famalicão/Barcelos, OCC, DMR, Modelo 22, IES, Sage). Meets mandatory requirements and fits the "weak, borderline" label well (adjacent degree in Public Administration & Management, routine in-house accounting, no Primavera/PHC or consultancy experience). Clean PT-PT and no stray cues; only minor quirk is the ISO date format (2022-09).

<details><summary>Text</summary>

# Matheus Siqueira

Residente em Portugal · Nacionalidade: Brasileira
Autorização para trabalhar em Portugal: sim (não necessita de visto)
Português: proficiência de nível nativo (C2)
Outras línguas: Inglês (C1)

## Perfil
**Contabilista Certificado (OCC)**

Contabilista com inscrição ativa na Ordem dos Contabilistas Certificados (OCC) e três anos de experiência profissional em contabilidade interna de empresas. Com experiência no lançamento de documentos contabilísticos de rotina ao abrigo do Sistema de Normalização Contabilística (SNC), na realização de reconciliações bancárias e na submissão de declarações fiscais regulares para uma empresa do setor de retalho.

## Experiência
**Contabilista Interno**, Comércio Têxtil do Ave, Lda., Vila Nova de Famalicão, 2022-09–Presente
- Responsável pela contabilidade geral da empresa de acordo com as normas do SNC.
- Classificação e registo de faturas de fornecedores, recibos de vendas e mapas de despesas de pessoal no Sage.
- Realização de reconciliações bancárias mensais e conferência de saldos de caixa.
- Preparação e submissão de declarações periódicas de IVA e declarações mensais de DMR e Segurança Social.
- Apoio no encerramento de contas anuais e na submissão da Modelo 22 e da IES.

**Assistente de Contabilidade Júnior**, Cerâmicas do Cávado, S.A., Barcelos, 2021-03–2022-08
- Organização, triagem e arquivo de documentação contabilística física e digital.
- Introdução de faturas de compra e notas de crédito no software de contabilidade interno.
- Verificação de extratos de conta corrente de fornecedores e preparação de listagens de pagamentos de rotina para a gerência.
- Apoio ao contabilista responsável em tarefas administrativas durante os prazos de cumprimento de obrigações fiscais.

## Formação
Licenciatura em Administração Pública e Gestão, Instituto Politécnico do Cávado e do Ave (IPCA), 2020

## Competências
Contabilidade Geral (SNC), Inscrição ativa na Ordem dos Contabilistas Certificados (OCC), Cumprimento de obrigações fiscais (IVA, IRS e IRC), Reconciliações Bancárias, Software de Contabilidade Sage, Microsoft Excel, Portal das Finanças


</details>
