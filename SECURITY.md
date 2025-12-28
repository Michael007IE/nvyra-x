NVYRA-X SECURITY POLICY
EFFECTIVE DATE: December 28, 2025 STATUS: ACTIVE & ENFORCED

At nvyra-x, we regard the security of our reasoning engines, proprietary search algorithms, and model weights as paramount. Given the high-value nature of our intellectual property, we enforce a strict, zero-tolerance policy regarding the handling of security vulnerabilities.

1. REPORTING A VULNERABILITY
DO NOT FILE A GITHUB ISSUE. Publicly disclosing a security vulnerability—even if well-intentioned—is a violation of our security policy and may result in the immediate revocation of access and potential legal action.

If you believe you have discovered a vulnerability in the nvyra-x stack (including the Tokasaurus engine, Hydragen cache, or the PRM verifiers), you must report it immediately and exclusively via:

Email: security@nvyra-x.com Subject Line: [CRITICAL] Security Finding - [Component Name]

We aim to acknowledge all reports within 24 hours.

2. IN-SCOPE VULNERABILITIES (2026 AI THREAT MODEL)
We are specifically interested in reports covering the following vectors:

Model Exfiltration: Techniques that allow unauthorized extraction of model weights or proprietary KV cache states.

Inference Bypass: Methods to bypass the "Verifier/Critic" step to force the generation of unverified or unsafe outputs.

Prompt Injection / Jailbreaking: Adversarial inputs that cause the reasoning model to deviate from its system prompt or safety guardrails.

Side-Channel Leaks: Timing or resource-usage attacks against the inference engine that reveal details about the search tree depth or width.

Remote Code Execution (RCE): Any exploit allowing arbitrary code execution on the cluster controllers or inference nodes.

3. OUT OF SCOPE
Social engineering (phishing) of nvyra-x employees.

Physical security attacks against our data centers or GPU providers.

Denial of Service (DoS/DDoS) attacks that simply exhaust compute quotas.

4. RULES OF ENGAGEMENT (SAFE HARBOR)
nvyra-x will not pursue legal action against you for security research that strictly adheres to these guidelines:

Private Disclosure: You must not disclose the vulnerability to any third party or the public until nvyra-x has explicitly authorized it in writing.

Non-Destructive: You must not harm our infrastructure, degrade the performance of our inference clusters, or destroy data.

Data Privacy: You must not view, access, or modify data belonging to other users or the company beyond what is strictly necessary to demonstrate the vulnerability.

No Financial Extortion: You must not demand payment in exchange for withholding the vulnerability disclosure.

5. ENCRYPTED COMMUNICATION
For highly sensitive reports (e.g., involving zero-day exploits in the inference kernel), please encrypt your email using our PGP key:


-----BEGIN PGP PUBLIC KEY BLOCK-----
(Paste your PGP Public Key here. If you don't have one, generate one via GPG.)
(This signals to hackers that you take security seriously.)
-----END PGP PUBLIC KEY BLOCK-----
6. BOUNTY & RECOGNITION
nvyra-x does not operate a public bug bounty program at this time. However, we may, at our sole discretion, offer financial rewards or recognition for critical findings that prevent significant loss of IP or compute resources.
