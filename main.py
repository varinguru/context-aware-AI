import os
import json
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings

# Load environment variables
load_dotenv()

# Configure Gemini as the LLM
llm = GoogleGenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="models/gemini-2.5-flash",  # or use "gemini-1.5-pro" for production
)
Settings.llm = llm

# Initialize LlamaCloud Index
index = LlamaCloudIndex(
    name="chase_bank",
    api_key=os.getenv("LLAMACLOUD_API_KEY")
)


def query_rag(query: str) -> str:
    """Query the RAG system and return the retrieved context as text."""
    engine = index.as_query_engine(llm=llm)
    response = engine.query(query)

    # Build context from retrieved nodes
    context_parts = []
    for node in response.source_nodes:
        context_parts.append(node.node.get_content())

    return "\n\n".join(context_parts)


def validate_policy(policy: Dict[str, Any], required_fields: list[str]) -> None:
    """Validate that required policy fields are present and not None."""
    missing_fields = [field for field in required_fields if policy.get(field) is None]
    if missing_fields:
        raise ValueError(f"Missing required policy fields: {', '.join(missing_fields)}")


def llm_extract_posting_policy(policy_text: str) -> Dict[str, Any]:
    """Extract transaction posting order policy from raw policy text using the LLM."""

    extraction_prompt = f"""Extract the transaction posting order policy from the text below.
Return ONLY a valid JSON object (no markdown, no code blocks) with these fields:
- deposit_first: true if deposits are posted before withdrawals (boolean)
- authorization_order_types: list of transaction types that are posted in authorization order. Include: "wire", "debit", "online", "atm", "teller_cash" if mentioned (array of strings, or null if not mentioned)
- remaining_items_order: how remaining items are sorted after auth-order types. Values: "highest_to_lowest", "lowest_to_highest", "chronological" (string)
- daily_negative_balance_fee: daily fee for having a negative balance (number or null)

If a field is not explicitly found in the text, use null.

Policy Text:
{policy_text}

JSON Response:"""

    response = llm.complete(extraction_prompt)

    try:
        response_text = response.text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        policy = json.loads(response_text.strip())
        return policy
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse posting policy JSON: {e}\nResponse was: {response.text}")


def llm_extract_fee_policy(policy_text: str) -> Dict[str, Any]:
    """Extract overdraft fee policy from raw policy text using the LLM."""

    extraction_prompt = f"""Extract the overdraft fee policy from the text below.
Return ONLY a valid JSON object (no markdown, no code blocks) with these fields:
- per_transaction_fee: the fee charged per overdraft transaction (as a number)
- max_daily_fees: maximum number of overdraft fees per day (as a number, or null if not mentioned)
- max_daily_fee_amount: maximum total overdraft fee amount per day (as a number, or null if not mentioned)

IMPORTANT: If the text says to "refer to your Fee Schedule" and does not contain a specific dollar amount, return null for per_transaction_fee. Do not make up or guess any fee amounts.

If a field is not found in the text, use null.

Policy Text:
{policy_text}

JSON Response:"""

    response = llm.complete(extraction_prompt)

    try:
        response_text = response.text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        policy = json.loads(response_text.strip())
        return policy
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse fee policy JSON: {e}\nResponse was: {response.text}")


def apply_hybrid_posting_order(
    transactions: list[Dict[str, Any]],
    posting_policy: Dict[str, Any]
) -> list[Dict[str, Any]]:
    """
    Apply Chase's hybrid transaction posting order policy.

    Order:
    1. Deposits first (if deposit_first is true)
    2. Authorization-order types (wire, debit, online, atm, teller_cash)
    3. Remaining items by specified order (highest_to_lowest, etc.)
    """
    deposit_first = posting_policy.get("deposit_first", False)
    auth_order_types = set(posting_policy.get("authorization_order_types", []))
    remaining_order = posting_policy.get("remaining_items_order", "highest_to_lowest")

    # Split transactions into categories
    deposits = []
    auth_order_txs = []
    remaining_txs = []

    for tx in transactions:
        tx_type = tx.get("transaction_type", "").lower()
        tx_amount = tx.get("amount", 0)

        if tx_amount > 0:
            deposits.append(tx)
        elif tx_type in auth_order_types:
            auth_order_txs.append(tx)
        else:
            remaining_txs.append(tx)

    # Sort remaining items according to policy
    if remaining_order == "highest_to_lowest":
        remaining_txs.sort(key=lambda t: abs(t.get("amount", 0)), reverse=True)
    elif remaining_order == "lowest_to_highest":
        remaining_txs.sort(key=lambda t: abs(t.get("amount", 0)))
    # chronological = already in order

    # Combine in posting order
    if deposit_first:
        sorted_transactions = deposits + auth_order_txs + remaining_txs
    else:
        sorted_transactions = auth_order_txs + remaining_txs + deposits

    return sorted_transactions


def calculate_overdraft(
    user_input: Dict[str, Any],
    posting_policy: Dict[str, Any],
    fee_policy: Dict[str, Any]
) -> str:
    """Calculate overdraft fees based on user input and extracted policies."""

    account_balance = user_input.get("account_balance", 0.0)
    transactions = user_input.get("transactions", [])
    overdraft_limit = user_input.get("overdraft_limit", 0.0)

    # Validate required fee fields
    per_transaction_fee = fee_policy.get("per_transaction_fee")
    if per_transaction_fee is None:
        raise ValueError(
            "Per-transaction fee not found in policy. The document may reference a "
            "separate Fee Schedule. Please provide the fee amount directly or ensure "
            "the fee schedule is indexed in RAG."
        )

    max_daily_fees = fee_policy.get("max_daily_fees")
    max_daily_fee_amount = fee_policy.get("max_daily_fee_amount")
    daily_negative_balance_fee = posting_policy.get("daily_negative_balance_fee")

    # Apply hybrid posting order
    sorted_transactions = apply_hybrid_posting_order(transactions, posting_policy)

    # Group transactions by date for daily fee limits
    from collections import defaultdict
    transactions_by_date = defaultdict(list)
    for tx in sorted_transactions:
        tx_date = tx.get("date", datetime.now().strftime("%Y-%m-%d"))
        transactions_by_date[tx_date].append(tx)

    # Process each day's transactions
    all_events = []
    current_balance = account_balance
    daily_negative_fee_applied = set()  # Track dates where daily fee was applied

    for date, day_txs in transactions_by_date.items():
        daily_fee_count = 0
        daily_total_fees = 0.0

        for tx in day_txs:
            tx_amount = tx.get("amount", 0)
            tx_description = tx.get("description", "")

            # Check if this transaction would cause overdraft
            new_balance = current_balance + tx_amount
            is_overdraft = new_balance < -overdraft_limit

            if is_overdraft and per_transaction_fee:
                # Check daily limits
                fee_to_apply = per_transaction_fee

                if max_daily_fees and daily_fee_count >= max_daily_fees:
                    fee_to_apply = 0.0
                elif max_daily_fee_amount and (daily_total_fees + fee_to_apply) > max_daily_fee_amount:
                    fee_to_apply = max_daily_fee_amount - daily_total_fees
                    if fee_to_apply < 0:
                        fee_to_apply = 0.0

                if fee_to_apply > 0:
                    daily_fee_count += 1
                    daily_total_fees += fee_to_apply
                    all_events.append({
                        "date": date,
                        "type": "overdraft_fee",
                        "amount": -fee_to_apply,
                        "description": f"Overdraft fee for transaction: {tx_description}"
                    })
                    new_balance -= fee_to_apply

            all_events.append({
                "date": date,
                "type": "transaction",
                "amount": tx_amount,
                "description": tx_description,
                "overdraft": is_overdraft
            })

            current_balance = new_balance

        # Apply daily negative balance fee at end of day
        if daily_negative_balance_fee and current_balance < 0 and date not in daily_negative_fee_applied:
            all_events.append({
                "date": date,
                "type": "daily_negative_balance_fee",
                "amount": -daily_negative_balance_fee,
                "description": f"Daily negative balance fee"
            })
            current_balance -= daily_negative_balance_fee
            daily_negative_fee_applied.add(date)

    # Build human-readable output
    output_lines = [
        "=== Overdraft Calculation Results ===",
        f"Starting Balance: ${account_balance:.2f}",
        f"Overdraft Limit: ${overdraft_limit:.2f}",
        f"Per-Transaction Fee: ${per_transaction_fee:.2f}",
        f"Deposit First: {posting_policy.get('deposit_first', 'N/A')}",
        f"Auth-Order Types: {', '.join(posting_policy.get('authorization_order_types', [])) or 'None'}",
        f"Remaining Items Order: {posting_policy.get('remaining_items_order', 'N/A')}",
        f"Daily Negative Balance Fee: ${daily_negative_balance_fee or 0:.2f}",
        f"Max Daily Fees: {max_daily_fees or 'Unlimited'}",
        f"Max Daily Fee Amount: ${max_daily_fee_amount or 'Unlimited'}",
        "",
        "--- Transaction Summary ---"
    ]

    total_fees = 0.0
    overdraft_count = 0

    for event in all_events:
        if event["type"] == "overdraft_fee":
            total_fees += abs(event["amount"])
            overdraft_count += 1
            output_lines.append(
                f"  {event['date']}: {event['description']} = ${event['amount']:.2f}"
            )
        elif event["type"] == "daily_negative_balance_fee":
            total_fees += abs(event["amount"])
            output_lines.append(
                f"  {event['date']}: {event['description']} = ${event['amount']:.2f}"
            )
        elif event.get("overdraft"):
            output_lines.append(
                f"  {event['date']}: {event['description']} = ${event['amount']:.2f} (OVERDRAFT)"
            )
        else:
            output_lines.append(
                f"  {event['date']}: {event['description']} = ${event['amount']:.2f}"
            )

    output_lines.extend([
        "",
        "--- Summary ---",
        f"Total Overdraft Fees: ${total_fees:.2f}",
        f"Number of Overdraft Events: {overdraft_count}",
        f"Final Balance: ${current_balance:.2f}"
    ])

    return "\n".join(output_lines)


def overdraft_tool(user_input: Dict[str, Any]) -> str:
    """
    Main orchestrator function for overdraft calculation.

    Args:
        user_input: Dictionary containing:
            - account_balance: Starting account balance
            - transactions: List of transaction dicts with:
                - amount: Transaction amount (negative for withdrawals, positive for deposits)
                - transaction_type: Transaction type for posting order (e.g., "wire", "debit", "online", "atm", "teller_cash", "check")
                - description: Transaction description (optional)
                - date: Transaction date (optional, defaults to today)
            - overdraft_limit: Overdraft protection limit (optional, defaults to 0)

    Returns:
        Human-readable text with overdraft calculation results
    """
    # Step 1: Query RAG for posting policy (separate query for precision)
    posting_policy_text = query_rag(
        "How does Chase determine transaction posting order for overdrafts? "
        "Include details about deposits, authorization order, and remaining items."
    )
    posting_policy = llm_extract_posting_policy(posting_policy_text)

    # Step 2: Query RAG for fee policy (separate query for precision)
    fee_policy_text = query_rag(
        "What fees apply when an account is overdrawn? Include per-transaction fees, "
        "daily limits, and any daily negative balance fees."
    )
    fee_policy = llm_extract_fee_policy(fee_policy_text)

    # Step 3: Validate required posting policy fields
    validate_policy(
        posting_policy,
        ["deposit_first", "authorization_order_types", "remaining_items_order"]
    )

    # Step 4: Calculate overdraft with both policies
    return calculate_overdraft(user_input, posting_policy, fee_policy)


# Example usage
if __name__ == "__main__":
    # Test the overdraft tool with transaction types
    test_input = {
        "account_balance": 100.00,
        "overdraft_limit": 100.00,
        "transactions": [
            {"amount": 200.00, "transaction_type": "deposit", "description": "Paycheck deposit", "date": "2025-01-15"},
            {"amount": -150.00, "transaction_type": "check", "description": "Rent payment", "date": "2025-01-15"},
            {"amount": -25.00, "transaction_type": "debit", "description": "Grocery store", "date": "2025-01-15"},
            {"amount": -50.00, "transaction_type": "atm", "description": "ATM withdrawal", "date": "2025-01-15"},
        ]
    }

    result = overdraft_tool(test_input)
    print(result)