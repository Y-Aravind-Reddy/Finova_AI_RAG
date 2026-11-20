"""Generate the ten FICTIONAL sample policy PDFs (two per role) under data/.

The documents are synthetic training material for "Finova Demo Bank", a
fictional institution. They are NOT official bank, RBI or other regulatory
policy, and every figure in them is invented for testing retrieval.

Uses a tiny built-in PDF writer so no extra dependency is needed.

Usage:  python scripts/generate_sample_pdfs.py [--force]
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DISCLAIMER = (
    "FICTIONAL TRAINING EXAMPLE - NOT an official bank, RBI or regulatory policy document. "
    "All names, figures and procedures are invented for software testing."
)

# role folder -> list of (filename, title, pages[ paragraphs[] ])
SAMPLES: dict[str, list[tuple[str, str, list[list[str]]]]] = {
    "super_admin": [
        (
            "platform_governance_policy.pdf",
            "Finova Demo Bank - Platform Access Governance Policy (Fictional)",
            [
                [
                    "1. Purpose. This fictional policy describes how privileged access to the Finova demo platform is granted, reviewed and revoked.",
                    "2. Role assignment. Roles are assigned only in the identity system profile table. A role change requires a written request from the tenant owner and approval by two Super Admins (two-person rule). Self-approval is not permitted.",
                    "3. Super Admin accounts. The fictional platform keeps no more than four active Super Admin accounts. Each Super Admin account must use hardware-key multi-factor authentication.",
                ],
                [
                    "4. Access reviews. Super Admins perform a privileged access review every 90 days. Accounts unused for 45 days are deactivated by setting is_active to false.",
                    "5. Break-glass access. Emergency break-glass credentials are sealed, used only during a declared Severity 1 incident, and rotated within 24 hours of use.",
                    "6. Audit. All role changes are written to an append-only audit log retained for 7 years in this fictional example.",
                ],
            ],
        ),
        (
            "incident_response_policy.pdf",
            "Finova Demo Bank - Security Incident Response Policy (Fictional)",
            [
                [
                    "1. Severity levels. Severity 1: confirmed exposure of customer data or full platform outage. Severity 2: partial outage or suspected data exposure. Severity 3: degraded service with no data risk.",
                    "2. Escalation. A Severity 1 incident must be escalated to the on-call Super Admin within 15 minutes and to the fictional Chief Risk Officer within 1 hour.",
                ],
                [
                    "3. Containment. The incident commander may suspend affected tenants, revoke sessions and disable API keys. Evidence must be preserved before systems are rebuilt.",
                    "4. Post-incident review. A blameless post-incident review is completed within 5 business days for Severity 1 and 2 incidents.",
                    "5. Tenant notification. Affected tenants are notified by the platform team within 24 hours of confirmation in this fictional policy.",
                ],
            ],
        ),
    ],
    "admin": [
        (
            "user_administration_policy.pdf",
            "Finova Demo Bank - User Administration Policy (Fictional)",
            [
                [
                    "1. Onboarding. Admins create user accounts only after HR confirmation of employment. New staff receive the least-privileged role needed for their job.",
                    "2. Offboarding. When an employee leaves, the Admin sets the profile is_active flag to false on the last working day, before the end of business.",
                ],
                [
                    "3. Password and MFA resets. Admins may reset multi-factor authentication only after verifying the employee through a video call and their manager's confirmation.",
                    "4. Role change requests. Admins submit role change requests to Super Admins; Admins cannot grant the Super Admin or Admin role themselves.",
                    "5. Shared accounts. Shared or generic user accounts are prohibited in this fictional policy.",
                ],
            ],
        ),
        (
            "operational_controls_policy.pdf",
            "Finova Demo Bank - Operational Controls Policy (Fictional)",
            [
                [
                    "1. Maker-checker. Any configuration change affecting fees, limits or interest rates needs a maker and a separate checker. The checker must hold the Admin role.",
                    "2. Change windows. Production changes are made in the fictional weekly change window, Tuesday 22:00 to 02:00, unless approved as an emergency change.",
                ],
                [
                    "3. Daily controls. Admins review the failed-login report and the suspended-transactions queue each business day by 11:00.",
                    "4. Data exports. Bulk exports of customer data require a ticket, a business justification and approval from the tenant data owner. Exports expire after 7 days.",
                ],
            ],
        ),
    ],
    "finance_analyst": [
        (
            "credit_risk_assessment_guidelines.pdf",
            "Finova Demo Bank - Credit Risk Assessment Guidelines (Fictional)",
            [
                [
                    "1. Scope. These fictional guidelines describe how analysts score retail personal loan applications in the demo environment.",
                    "2. Internal score bands. Band A: internal score 750 and above. Band B: 650 to 749. Band C: 550 to 649. Band D: below 550. These bands are invented and do not correspond to any credit bureau.",
                    "3. Debt-to-income. The fictional maximum debt-to-income ratio for a Band B applicant is 40 percent of verified monthly income.",
                ],
                [
                    "4. Exceptions. Any approval outside the band rules is an exception and must be signed off by a senior credit analyst with a written rationale.",
                    "5. Portfolio review. Analysts review the personal loan portfolio monthly and flag any band whose 90-day delinquency exceeds the fictional trigger of 3 percent.",
                ],
            ],
        ),
        (
            "treasury_and_reporting_policy.pdf",
            "Finova Demo Bank - Treasury and Management Reporting Policy (Fictional)",
            [
                [
                    "1. Liquidity buffer. In this fictional example, treasury keeps an internal liquidity buffer target equal to 20 days of projected net outflows.",
                    "2. Reporting calendar. The monthly management information pack is finalised by the 7th business day of the following month.",
                ],
                [
                    "3. Reconciliation. General ledger to sub-ledger reconciliations are completed daily; breaks older than 3 business days are escalated to the finance controller.",
                    "4. Data quality. Analysts must not alter source data in reports. Adjustments are recorded as separate, documented journal entries.",
                ],
            ],
        ),
    ],
    "support_executive": [
        (
            "complaint_handling_procedure.pdf",
            "Finova Demo Bank - Complaint Handling Procedure (Fictional)",
            [
                [
                    "1. Logging. Every complaint is logged in the ticketing system with a reference number given to the customer at first contact.",
                    "2. Acknowledgement. Support executives acknowledge complaints within 1 business day in this fictional procedure.",
                    "3. Resolution target. The internal resolution target is 10 business days. Complex cases may be extended once, with the customer informed in writing.",
                ],
                [
                    "4. Escalation. Complaints involving suspected fraud are escalated immediately to the fraud team and are not discussed further on the call.",
                    "5. Closure. A complaint is closed only after the customer has been told the outcome and the reason for it.",
                ],
            ],
        ),
        (
            "customer_verification_procedure.pdf",
            "Finova Demo Bank - Customer Verification Procedure (Fictional)",
            [
                [
                    "1. Verification before disclosure. Before discussing any account, the executive verifies the caller with a one-time passcode sent to the registered mobile number plus two knowledge questions.",
                    "2. Never ask for secrets. Executives must never ask for a full card number, CVV, PIN, password or one-time passcode used for payments.",
                ],
                [
                    "3. Failed verification. After 3 failed verification attempts the call is ended politely and the account is flagged for review.",
                    "4. Third parties. Information is never disclosed to a third party unless a valid authorisation is recorded on the account.",
                ],
            ],
        ),
    ],
    "customer": [
        (
            "account_terms_and_fees.pdf",
            "Finova Demo Bank - Savings Account Terms and Fees (Fictional)",
            [
                [
                    "1. About this document. These are fictional terms for a demo savings account. They are not an offer and do not describe any real bank product.",
                    "2. Minimum balance. The fictional Finova Demo Savings account has no minimum balance requirement.",
                    "3. Fees. Domestic online transfers are free. A replacement debit card costs a fictional fee of 150 demo credits.",
                ],
                [
                    "4. Account closure. To close an account, submit a closure request in the app or at a branch. Remaining funds are transferred within 3 business days to an account you nominate.",
                    "5. Statements. Monthly e-statements are available in the app for 24 months.",
                ],
            ],
        ),
        (
            "digital_banking_safety_guide.pdf",
            "Finova Demo Bank - Digital Banking Safety Guide (Fictional)",
            [
                [
                    "1. We never ask for your PIN. Finova Demo Bank staff will never ask for your PIN, password or one-time passcode by phone, email or message.",
                    "2. Lost phone or card. Block your card instantly from the app, or call the fictional 24x7 helpline listed in the app.",
                ],
                [
                    "3. Suspicious transactions. Report unauthorised transactions in the app as soon as possible. Reporting quickly helps the bank investigate.",
                    "4. Device safety. Keep your phone's operating system and the banking app updated, and do not install apps from unknown sources.",
                ],
            ],
        ),
    ],
}


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _page_stream(title: str, paragraphs: list[str], page_no: int, page_count: int) -> bytes:
    lines: list[tuple[str, int, str]] = [("F2", 13, title), ("F1", 1, "")]
    for line in textwrap.wrap(DISCLAIMER, 95):
        lines.append(("F2", 9, line))
    lines.append(("F1", 1, ""))
    for para in paragraphs:
        for line in textwrap.wrap(para, 92):
            lines.append(("F1", 10, line))
        lines.append(("F1", 1, ""))
    lines.append(("F1", 8, f"Page {page_no} of {page_count} - Fictional sample document"))

    ops = ["BT", "50 790 Td"]
    for font, size, text in lines:
        if not text:
            ops.append("0 -8 Td")
            continue
        ops.append(f"/{font} {size} Tf")
        ops.append(f"({_escape(text)}) Tj")
        ops.append(f"0 -{size + 5} Td")
    ops.append("ET")
    return "\n".join(ops).encode("latin-1")


def write_pdf(path: Path, title: str, pages: list[list[str]]) -> None:
    """Write a simple multi-page text PDF with valid xref offsets."""
    objects: list[bytes] = []
    n_pages = len(pages)
    # 1 catalog, 2 pages, 3 font regular, 4 font bold, then (page, content) pairs, then info
    page_ids = [5 + 2 * i for i in range(n_pages)]
    info_id = 5 + 2 * n_pages
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    for i, paragraphs in enumerate(pages):
        content_id = page_ids[i] + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        stream = _page_stream(title, paragraphs, i + 1, n_pages)
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(f"<< /Title ({_escape(title)}) /Subject (Fictional training example) /Producer (FINOVA sample generator) >>".encode())

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {info_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Overwrite existing sample PDFs.")
    args = parser.parse_args(argv)
    created = 0
    for folder, docs in SAMPLES.items():
        target = DATA / folder
        target.mkdir(parents=True, exist_ok=True)
        for filename, title, pages in docs:
            path = target / filename
            if path.exists() and not args.force:
                print(f"exists, skipped: {path.relative_to(ROOT)}")
                continue
            write_pdf(path, title, pages)
            created += 1
            print(f"created: {path.relative_to(ROOT)}")
    print(f"{created} PDF(s) written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
