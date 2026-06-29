from pathlib import Path
part1 = open("/home/linyu/桌面/se agent/model_brain_surgery/_p1.md").read()
part2 = open("/home/linyu/桌面/se agent/model_brain_surgery/_p2.md").read()
Path("/home/linyu/桌面/se agent/model_brain_surgery/README.md").write_text(part1 + part2)
print("OK")
