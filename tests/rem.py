from conduit.remote import RemoteModelSync

model = RemoteModelSync(model="gpt-oss:latest")
result = model.query("Name ten mammals")
