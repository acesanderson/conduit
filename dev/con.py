from conduit.sync import Conduit, Model

m = Model("gpt3")
# m.options.cache.wipe()
r = m.query("Hello, world!")
print(r)

"""
f7d93e57498960c17c7cf2edcda41bbc5779235228b8bc522a73903c1c0b90e4
4c796781437d753832261c658852be20ac062ed1b73bb2f3e2bc0739c84dc9df
"""
