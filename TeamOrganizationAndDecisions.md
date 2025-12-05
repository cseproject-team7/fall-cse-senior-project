# Team Organization and Decisions

This document outlines our team structure, decision-making methodology, and the key architectural choices made during the development of our senior project.

## 1. Team Composition
The following members comprise the project team:

1. **Manav Viramgama**
2. **Jyothik Addala**
3. **Ishraq Khan**
4. **Aleksandr Klochkov**

## 2. Decision Making Process

### Agile Methodology
We have adopted an Agile approach to development:
* **MVP First:** We prioritize building a Minimum Viable Product (MVP) first.
* **Scaling:** Once the MVP is stable, we scale the solution to handle larger amounts of logs and data volume.

### Conflict Resolution
When disagreements arise regarding implementation or strategy:
* **Proposal Phase:** All proposed solutions are explained in detail to the group.
* **Voting:** The team votes on the options to decide which solution to implement, ensuring a democratic process.

## 3. Key Architectural and Organizational Decisions

| Chosen Architecture | Alternative Considered | Reasoning |
| :--- | :--- | :--- |
| **Simulated User Logs** | Real User Logs | Due to security concerns, we were not able to gain access to USF's real user logs. |
| **Node.js** | Python | We chose Node.js for the backend because the heavy computation is offloaded to Azure. Since the backend does not require heavy data processing (which Python excels at), Node.js is more efficient for simply retrieving required information as needed. |
| **ReactJS** | AngularJS | We selected ReactJS over AngularJS because the team has more prior experience with React. The smaller learning curve allows us to shift focus to other critical components of the project rather than learning a new framework. |
| **Azure ML** | Backend ML (Local) | The machine learning model is hosted on Azure for speed, storage, and easier environment management. Hosting locally would increase deployment complexity. *Note: We train/test locally to save costs, then host the final model on Azure.* |
| **Unstructured Data Storage** | Structured Data Storage | We opted for unstructured storage (Azure Data Lake Storage) because the project does not require complex querying of the data, only retrieval and usage. |
