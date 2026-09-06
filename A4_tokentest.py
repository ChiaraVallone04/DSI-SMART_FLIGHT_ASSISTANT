import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

consulta_es = (
    "Sos un asistente de viajes. Decime cuánto cuesta volar de Madrid (MAD) a Berlín (BER) "
    "para 2 personas a mediados de octubre de 2026 y cuándo me conviene comprar el pasaje."
)

consulta_en = (
    "You are a travel assistant. Tell me how much it costs to fly from Madrid (MAD) to Berlin (BER) "
    "for 2 people in mid-October 2026 and when I should buy the ticket."
)

print(f"ES: {len(enc.encode(consulta_es))} tokens")
print(f"EN: {len(enc.encode(consulta_en))} tokens")