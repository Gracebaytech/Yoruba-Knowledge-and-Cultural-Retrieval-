import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime


load_dotenv()


# Read signing key from the environment; do NOT exit on missing key so the
# module can be imported by `uvicorn main:app` during local development.
signing_key = "22dc8f8f9534166b57aac7048efef00c06672ee034595c3a450cbf5132bd703e"
if not signing_key:
    print("WARNING: INNGEST_SIGNING_KEY not set. Inngest endpoints will be disabled.")
    print("To enable Inngest features, set INNGEST_SIGNING_KEY in your environment or .env file.")
    inngest_enabled = False
else:
    inngest_enabled = True

# Create an Inngest client (signing_key may be None for local dev)
inngest_client = inngest.Inngest(
    app_id="yoruba-rag",
    logger=logging.getLogger("unicorn.inngest "),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
    signing_key=signing_key,
)

# Create an Inngest function
@inngest_client.create_function(
    fn_id="yoruba_rag",
    # Event that triggers this function
    trigger=inngest.TriggerEvent(event="app/yoruba_rag"),
)
async def my_function(ctx: inngest.Context) -> str:
    ctx.logger.info(ctx.event)
    return "done"

app = FastAPI()

# Only wire the Inngest FastAPI handler if a signing key is present.
if inngest_enabled:
    try:
        inngest.fast_api.serve(app, inngest_client, [my_function])
    except Exception as exc:
        print("Warning: failed to start Inngest FastAPI handler:", exc)
        print("Proceeding without Inngest endpoints.")
else:
    print("Inngest disabled: skipping Inngest FastAPI setup")




