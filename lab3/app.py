"""LearnPath AI - 基于知识图谱的自适应编程学习平台 (原型)."""

import json
import random
import os
import re
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
load_dotenv(override=True)

app = Flask(__name__)

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_MODEL = 'deepseek-v4-flash'

REVIEW_PROMPT = """指出这段 Python 代码的问题和改进建议。每行一条反馈，前缀 [warning] 或 [tip] 或 [info]。末尾写 评分=XX/100。

{code}"""

# 知识图谱: Python 入门概念及其依赖关系
KNOWLEDGE_GRAPH = {
    "变量与数据类型": {"prereqs": [], "difficulty": 1, "category": "基础语法"},
    "运算符与表达式": {"prereqs": ["变量与数据类型"], "difficulty": 1, "category": "基础语法"},
    "条件判断": {"prereqs": ["变量与数据类型", "运算符与表达式"], "difficulty": 2, "category": "控制流"},
    "循环结构": {"prereqs": ["条件判断"], "difficulty": 2, "category": "控制流"},
    "列表与元组": {"prereqs": ["变量与数据类型"], "difficulty": 2, "category": "数据结构"},
    "字符串处理": {"prereqs": ["变量与数据类型"], "difficulty": 2, "category": "数据结构"},
    "字典与集合": {"prereqs": ["列表与元组"], "difficulty": 3, "category": "数据结构"},
    "函数定义": {"prereqs": ["条件判断", "循环结构", "列表与元组"], "difficulty": 3, "category": "函数"},
    "函数参数与返回值": {"prereqs": ["函数定义"], "difficulty": 3, "category": "函数"},
    "文件读写": {"prereqs": ["字符串处理", "字典与集合"], "difficulty": 4, "category": "文件操作"},
    "异常处理": {"prereqs": ["函数定义", "文件读写"], "difficulty": 4, "category": "高级特性"},
    "面向对象基础": {"prereqs": ["函数定义", "字典与集合"], "difficulty": 4, "category": "OOP"},
    "模块与包": {"prereqs": ["函数定义", "文件读写"], "difficulty": 4, "category": "高级特性"},
    "列表推导式": {"prereqs": ["列表与元组", "循环结构"], "difficulty": 3, "category": "数据结构"},
}

# 练习题
EXERCISES = {
    # 1级
    1: [
        {"q": "以下哪个是正确的变量赋值？", "opts": ["x = 10", "int x = 10", "var x = 10", "x := 10"], "ans": 0, "concept": "变量与数据类型", "hint": "Python 不需要类型声明"},
        {"q": "3 + 5 * 2 的结果是多少？", "opts": ["16", "13", "10", "8"], "ans": 1, "concept": "运算符与表达式", "hint": "乘除优先级高于加减"},
        {"q": "字符串 'hello' + ' world' 的结果是？", "opts": ["hello world", "helloworld", "错误", "hello + world"], "ans": 0, "concept": "变量与数据类型", "hint": "+ 可以拼接字符串"},
    ],
    # 2级
    2: [
        {"q": "for i in range(3): print(i) 会输出什么？", "opts": ["1 2 3", "0 1 2", "0 1 2 3", "1 2"], "ans": 1, "concept": "循环结构", "hint": "range(n) 从 0 到 n-1"},
        {"q": "if x > 5: 中，哪个 x 值会使条件为 True？", "opts": ["x = 3", "x = 5", "x = 7", "x = 0"], "ans": 2, "concept": "条件判断", "hint": "> 表示大于，不包含等于"},
        {"q": "lst = [1, 2, 3]; lst.append(4); 后 lst 是？", "opts": ["[1, 2, 3]", "[4, 1, 2, 3]", "[1, 2, 3, 4]", "错误"], "ans": 2, "concept": "列表与元组", "hint": "append 在末尾添加元素"},
    ],
    # 3级
    3: [
        {"q": "def add(a, b=1): return a + b; add(3) 返回？", "opts": ["4", "错误", "1", "3"], "ans": 0, "concept": "函数参数与返回值", "hint": "b=1 是默认参数值"},
        {"q": "d = {'a': 1}; d['b'] = 2; 后 d 的值是？", "opts": ["{'a': 1}", "{'a': 1, 'b': 2}", "错误", "{'b': 2}"], "ans": 1, "concept": "字典与集合", "hint": "可以直接通过键添加新元素"},
        {"q": "[x*2 for x in range(3)] 输出？", "opts": ["[0, 1, 2]", "[0,2,4]", "[1,2,3]", "错误"], "ans": 1, "concept": "列表推导式", "hint": "列表推导式对每个元素执行表达式"},
    ],
    # 4级
    4: [
        {"q": "class Dog: pass; d = Dog(); d.name = 'Buddy' 中 d.name 是什么？", "opts": ["None", "错误", "'Buddy'", "Dog"], "ans": 2, "concept": "面向对象基础", "hint": "Python 可以动态添加属性"},
        {"q": "try: x=1/0; except ZeroDivisionError: print('err') 输出？", "opts": ["0", "err", "程序崩溃", "None"], "ans": 1, "concept": "异常处理", "hint": "except 捕获指定的异常类型"},
    ],
}

# 代码审查规则
CODE_REVIEW_RULES = [
    {"pattern": "==", "check": lambda code: "==" in code and "if" in code, "msg": "使用了 == 比较，正确！", "level": "info"},
    {"pattern": "=", "check": lambda code: " =" in code and "if " in code and " ==" not in code, "msg": "条件判断中用 = 赋值而非 == 比较？Python 不支持在 if 中赋值。", "level": "warning"},
    {"pattern": "range", "check": lambda code: "range(" in code, "msg": "range() 是 Python 内置的迭代器，用于生成数字序列", "level": "tip"},
    {"pattern": "def", "check": lambda code: "def " in code and ":" not in code.split("def ")[-1].split("(")[-1].split(")")[-1].strip(), "msg": "函数定义末尾需要冒号 :", "level": "warning"},
    {"pattern": "indent", "check": lambda code: "\t" in code, "msg": "代码中使用了 Tab 缩进，Python 建议使用 4 个空格", "level": "tip"},
    {"pattern": "f-string", "check": lambda code: "print(f" in code, "msg": "使用了 f-string 格式化，这是 Python 3.6+ 推荐的字符串格式化方式", "level": "tip"},
    {"pattern": "list comp", "check": lambda code: "[" in code and "for " in code and "]" in code and "[" in code.split("for")[0], "msg": "使用了列表推导式，简洁高效！", "level": "info"},
    {"pattern": "input", "check": lambda code: "input(" in code, "msg": "使用 input() 获取用户输入，返回值始终是字符串类型", "level": "tip"},
]

# 学生状态存储 (模拟)
student_profiles = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/knowledge-graph')
def knowledge_graph():
    """返回知识图谱数据."""
    nodes = []
    edges = []
    for concept, info in KNOWLEDGE_GRAPH.items():
        nodes.append({
            "id": concept,
            "label": concept,
            "category": info["category"],
            "difficulty": info["difficulty"],
            "group": info["category"]
        })
        for pre in info["prereqs"]:
            edges.append({"from": pre, "to": concept})

    return jsonify({"nodes": nodes, "edges": edges})


@app.route('/api/exercise/<int:difficulty>')
def get_exercise(difficulty):
    """根据难度获取练习题."""
    exercises = EXERCISES.get(difficulty, EXERCISES[1])
    ex = random.choice(exercises)
    return jsonify(ex)


@app.route('/api/review-code', methods=['POST'])
def review_code():
    """AI 代码审查 (Ollama qwen3.5:0.8b)."""
    code = request.json.get('code', '')

    if not code.strip():
        return jsonify({
            "feedbacks": [{"msg": "未检测到代码", "level": "warning"}],
            "score": 0, "source": "rule"
        })

    # 尝试 DeepSeek API 审查
    if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != 'your_api_key_here':
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url='https://api.deepseek.com/v1')

            prompt = REVIEW_PROMPT.format(code=code)
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                stream=False, timeout=60)

            raw = resp.choices[0].message.content.strip()
            feedbacks = []
            score = 70
            for line in raw.split('\n'):
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                for prefix, level in [('[warning]', 'warning'), ('[tip]', 'tip'),
                                       ('[info]', 'info')]:
                    if line.lower().startswith(prefix):
                        msg = line[len(prefix):].lstrip('-:： ').strip()
                        if msg and len(msg) > 2:
                            feedbacks.append({"msg": msg, "level": level})
                        break
                m = re.search(r'评分\s*[=:：]\s*(\d+)', line)
                if m:
                    score = int(m.group(1))

            if feedbacks:
                return jsonify({
                    "feedbacks": feedbacks,
                    "score": min(100, max(0, score)),
                    "source": "deepseek"
                })
        except Exception as e:
            print(f"[DeepSeek 失败, 回退规则引擎]: {type(e).__name__}")

    # 回退: 规则引擎
    return _rule_based_review(code)


def _rule_based_review(code):
    """规则引擎代码审查 (回退方案)."""
    feedbacks = []
    score = 100

    for rule in CODE_REVIEW_RULES:
        if rule["check"](code):
            feedbacks.append({"msg": rule["msg"], "level": rule["level"]})
            if rule["level"] == "warning":
                score -= 15

    bonus = 0
    if "def " in code and ":" in code.split("def ")[-1].split(")")[-1].strip():
        bonus += 10
        feedbacks.append({"msg": "函数定义语法正确", "level": "info"})
    if "#" in code or '"""' in code:
        bonus += 5
        feedbacks.append({"msg": "有代码注释，良好的编程习惯！", "level": "tip"})

    score = max(0, min(100, score + bonus))
    if not feedbacks:
        feedbacks.append({"msg": "代码看起来不错！", "level": "info"})

    return jsonify({"feedbacks": feedbacks, "score": score, "source": "rule"})


@app.route('/api/learning-path', methods=['POST'])
def learning_path():
    """根据掌握情况推荐学习路径."""
    mastered = set(request.json.get('mastered', []))
    all_concepts = set(KNOWLEDGE_GRAPH.keys())

    # 找出可以学习的下一批概念 (前置都已掌握)
    available = []
    for concept in all_concepts - mastered:
        prereqs = set(KNOWLEDGE_GRAPH[concept]["prereqs"])
        if prereqs.issubset(mastered):
            available.append({
                "concept": concept,
                "difficulty": KNOWLEDGE_GRAPH[concept]["difficulty"],
                "category": KNOWLEDGE_GRAPH[concept]["category"]
            })

    # 按难度排序
    available.sort(key=lambda x: x["difficulty"])

    # 推荐路径
    path = []
    for item in available:
        path.append(f"{item['concept']} (难度: {'★' * item['difficulty']}{'☆' * (5 - item['difficulty'])}, {item['category']})")

    return jsonify({
        "available": available,
        "recommendations": path[:5],
        "mastered_count": len(mastered),
        "total_count": len(all_concepts)
    })


@app.route('/api/simulate-student')
def simulate_student():
    """生成模拟学生数据."""
    profiles = [
        {
            "name": "小明 (模拟学习者)",
            "mastered": ["变量与数据类型"],
            "strengths": "理解语法规则快，但缺乏问题分解能力",
            "weakness": "循环和函数理解困难",
            "recommended_next": "运算符与表达式",
            "learning_style": "动手实践型"
        },
        {
            "name": "小红 (模拟学习者)",
            "mastered": ["变量与数据类型", "运算符与表达式", "条件判断", "循环结构"],
            "strengths": "逻辑思维能力强",
            "weakness": "复杂数据结构和抽象概念",
            "recommended_next": "列表与元组",
            "learning_style": "理论理解型"
        }
    ]
    return jsonify(random.choice(profiles))


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  LearnPath AI - 自适应编程学习平台原型")
    print("  访问 http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
