# -*- coding: utf-8 -*-
"""
Markdown 转换工具模块
支持 Markdown 转 HTML

技术栈:
- Markdown 转 HTML: markdown-it-py + mdit-py-plugins + pygments
"""
import os
from pathlib import Path
from typing import Optional

from loguru import logger

# ===================== 依赖检测 =====================
try:
    from markdown_it import MarkdownIt
    from mdit_py_plugins.tasklists import tasklists_plugin
    from mdit_py_plugins.footnote import footnote_plugin
    from mdit_py_plugins.deflist import deflist_plugin
    from mdit_py_plugins.dollarmath import dollarmath_plugin
    MARKDOWN_IT_AVAILABLE = True
except ImportError:
    MARKDOWN_IT_AVAILABLE = False

try:
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters import HtmlFormatter
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False


# ===================== CSS 样式 =====================
DEFAULT_HTML_CSS = """
:root {
    --bg-color: #ffffff;
    --text-color: #24292e;
    --code-bg: #f6f8fa;
    --border-color: #e1e4e8;
    --link-color: #0366d6;
    --blockquote-color: #6a737d;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    color: var(--text-color);
    background-color: var(--bg-color);
    max-width: 900px;
    margin: 0 auto;
    padding: 20px 45px;
}

h1, h2, h3, h4, h5, h6 {
    margin-top: 24px;
    margin-bottom: 16px;
    font-weight: 600;
    line-height: 1.25;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: .3em;
}

h1 { font-size: 2em; }
h2 { font-size: 1.5em; }
h3 { font-size: 1.25em; border-bottom: none; }
h4, h5, h6 { border-bottom: none; }

p { margin-top: 0; margin-bottom: 16px; }

a { color: var(--link-color); text-decoration: none; }
a:hover { text-decoration: underline; }

code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 85%;
    background-color: var(--code-bg);
    padding: 0.2em 0.4em;
    border-radius: 6px;
}

pre {
    background-color: var(--code-bg);
    border-radius: 6px;
    padding: 16px;
    overflow: auto;
    font-size: 85%;
    line-height: 1.45;
}

pre code {
    background: transparent;
    padding: 0;
    font-size: 100%;
}

blockquote {
    margin: 0;
    padding: 0 1em;
    color: var(--blockquote-color);
    border-left: 0.25em solid var(--border-color);
}

table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 16px;
}

table th, table td {
    padding: 6px 13px;
    border: 1px solid var(--border-color);
}

table th {
    font-weight: 600;
    background-color: var(--code-bg);
}

table tr:nth-child(2n) {
    background-color: #f6f8fa;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 16px 0;
}

hr {
    height: 0.25em;
    padding: 0;
    margin: 24px 0;
    background-color: var(--border-color);
    border: 0;
}

/* 任务列表样式 */
ul.contains-task-list {
    list-style-type: none;
    padding-left: 0;
}

.task-list-item {
    list-style-type: none;
}

.task-list-item input[type="checkbox"] {
    margin-right: 8px;
}

/* 代码高亮样式 (Pygments) */
.highlight .hll { background-color: #ffffcc }
.highlight .c { color: #6a737d }
.highlight .k { color: #d73a49 }
.highlight .o { color: #24292e }
.highlight .cm { color: #6a737d }
.highlight .cp { color: #d73a49 }
.highlight .c1 { color: #6a737d }
.highlight .cs { color: #6a737d }
.highlight .gd { color: #b31d28; background-color: #ffeef0 }
.highlight .ge { font-style: italic }
.highlight .gi { color: #22863a; background-color: #f0fff4 }
.highlight .gs { font-weight: bold }
.highlight .gu { color: #6f42c1 }
.highlight .kc { color: #005cc5 }
.highlight .kd { color: #d73a49 }
.highlight .kn { color: #d73a49 }
.highlight .kp { color: #d73a49 }
.highlight .kr { color: #d73a49 }
.highlight .kt { color: #d73a49 }
.highlight .m { color: #005cc5 }
.highlight .s { color: #032f62 }
.highlight .na { color: #6f42c1 }
.highlight .nb { color: #005cc5 }
.highlight .nc { color: #6f42c1 }
.highlight .no { color: #005cc5 }
.highlight .nd { color: #6f42c1 }
.highlight .ni { color: #24292e }
.highlight .ne { color: #005cc5 }
.highlight .nf { color: #6f42c1 }
.highlight .nl { color: #005cc5 }
.highlight .nn { color: #6f42c1 }
.highlight .nt { color: #22863a }
.highlight .nv { color: #e36209 }
.highlight .ow { color: #d73a49 }
.highlight .w { color: #bbbbbb }
.highlight .mf { color: #005cc5 }
.highlight .mh { color: #005cc5 }
.highlight .mi { color: #005cc5 }
.highlight .mo { color: #005cc5 }
.highlight .sb { color: #032f62 }
.highlight .sc { color: #032f62 }
.highlight .sd { color: #032f62 }
.highlight .s2 { color: #032f62 }
.highlight .se { color: #032f62 }
.highlight .sh { color: #032f62 }
.highlight .si { color: #005cc5 }
.highlight .sx { color: #032f62 }
.highlight .sr { color: #032f62 }
.highlight .s1 { color: #032f62 }
.highlight .ss { color: #005cc5 }
.highlight .bp { color: #005cc5 }
.highlight .vc { color: #e36209 }
.highlight .vg { color: #e36209 }
.highlight .vi { color: #e36209 }
.highlight .il { color: #005cc5 }

/* 数学公式样式 */
.math { font-family: "Times New Roman", serif; }
.math-inline { display: inline; }
.math-block { 
    display: block; 
    text-align: center;
    margin: 1em 0;
}
/* MathJax 渲染的公式样式 */
.MathJax { 
    display: inline-block;
    margin: 0;
}
.MathJax_Display {
    display: block;
    margin: 1em 0;
    text-align: center;
}

/* 脚注样式 */
.footnote-ref { font-size: 0.8em; vertical-align: super; }
.footnotes { font-size: 0.9em; margin-top: 2em; border-top: 1px solid var(--border-color); padding-top: 1em; }

/* 定义列表样式 */
dl { margin: 16px 0; }
dt { font-weight: 600; margin-top: 16px; }
dd { margin-left: 16px; margin-bottom: 16px; }

/* 打印样式优化 */
@media print {
    body { max-width: none; padding: 20px; }
    pre, code { white-space: pre-wrap; word-wrap: break-word; }
}
"""


def _highlight_code(code: str, lang: str) -> str:
    """使用 Pygments 进行代码高亮"""
    if not PYGMENTS_AVAILABLE:
        return f'<pre><code class="language-{lang}">{code}</code></pre>'
    
    try:
        if lang:
            lexer = get_lexer_by_name(lang, stripall=True)
        else:
            lexer = guess_lexer(code)
        formatter = HtmlFormatter(nowrap=True, cssclass="highlight")
        highlighted = highlight(code, lexer, formatter)
        return f'<pre><code class="highlight language-{lang}">{highlighted}</code></pre>'
    except Exception:
        return f'<pre><code class="language-{lang}">{code}</code></pre>'


def _create_markdown_parser(enable_code_highlight: bool = True) -> "MarkdownIt":
    """
    创建配置完整的 Markdown 解析器
    支持: GFM表格、任务列表、脚注、定义列表、代码块高亮等
    """
    md = MarkdownIt("gfm-like", {"typographer": True, "html": True, "linkify": False})
    
    # 添加扩展插件
    md.use(tasklists_plugin)  # 任务列表支持 - [ ] / - [x]
    md.use(footnote_plugin)   # 脚注支持 [^1]
    md.use(deflist_plugin)    # 定义列表支持
    md.use(dollarmath_plugin, allow_space=True, allow_digits=True, double_inline=True)  # 数学公式支持 $...$ 和 $$...$$
    
    # 自定义数学公式渲染，使用 MathJax 兼容格式
    # 直接使用 $...$ 和 $$...$$ 格式，MathJax 默认支持
    def render_math_inline(renderer, tokens, idx, options, env):
        token = tokens[idx]
        # 使用 $...$ 格式，MathJax 会自动识别
        return f'<span class="math-inline">${token.content}$</span>'
    
    def render_math_block(renderer, tokens, idx, options, env):
        token = tokens[idx]
        # 使用 $$...$$ 格式，MathJax 会自动识别
        return f'<div class="math-block">$${token.content}$$</div>\n'
    
    md.add_render_rule("math_inline", render_math_inline)
    md.add_render_rule("math_block", render_math_block)
    
    # 自定义代码块渲染（如果启用高亮）
    if enable_code_highlight and PYGMENTS_AVAILABLE:
        def render_fence(renderer, tokens, idx, options, env):
            token = tokens[idx]
            lang = token.info.strip() if token.info else ""
            code = token.content
            return _highlight_code(code, lang)
        
        md.add_render_rule("fence", render_fence)
    
    return md


def markdown_to_html(
    markdown_content: str,
    output_path: Optional[str] = None,
    title: str = "Markdown Document",
    enable_code_highlight: bool = True,
    custom_css: Optional[str] = None,
    embed_images: bool = False,
    image_base_path: Optional[str] = None
) -> str:
    """
    将 Markdown 转换为 HTML
    
    Args:
        markdown_content: Markdown 文本内容
        output_path: 输出HTML文件路径，如果为None则只返回HTML字符串
        title: HTML文档标题
        enable_code_highlight: 是否启用代码高亮
        custom_css: 自定义CSS样式
        embed_images: 是否将图片嵌入为base64
        image_base_path: 图片基础路径，用于解析相对路径的图片
        
    Returns:
        str: 生成的HTML内容
    """
    if not MARKDOWN_IT_AVAILABLE:
        raise ImportError(
            "markdown-it-py 未安装，Markdown转HTML功能不可用。\n"
            "请运行: pip install markdown-it-py mdit-py-plugins pygments"
        )
    
    # 创建解析器并转换
    md = _create_markdown_parser(enable_code_highlight)
    html_body = md.render(markdown_content)
    
    # 处理图片嵌入（可选）
    if embed_images and image_base_path:
        import base64
        import re
        
        def embed_image(match):
            img_src = match.group(1)
            if img_src.startswith(('http://', 'https://', 'data:')):
                return match.group(0)
            
            img_path = os.path.join(image_base_path, img_src) if not os.path.isabs(img_src) else img_src
            if os.path.exists(img_path):
                try:
                    with open(img_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                    ext = os.path.splitext(img_path)[1].lower()
                    mime_map = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp', 'svg': 'svg+xml'}
                    mime_type = mime_map.get(ext.lstrip('.'), 'png')
                    return f'src="data:image/{mime_type};base64,{img_data}"'
                except Exception as e:
                    logger.warning(f"无法嵌入图片 {img_path}: {e}")
            return match.group(0)
        
        html_body = re.sub(r'src="([^"]+)"', embed_image, html_body)
    
    css_content = custom_css if custom_css else DEFAULT_HTML_CSS
    
    # 构建完整的HTML文档（包含 MathJax 用于渲染数学公式）
    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{css_content}
    </style>
    <!-- MathJax 配置，用于渲染 LaTeX 数学公式 -->
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true,
                processEnvironments: true
            }},
            options: {{
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
            }}
        }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
</head>
<body>
{html_body}
</body>
</html>"""
    
    # 保存到文件（如果指定了路径）
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        logger.info(f"HTML文档已保存到: {output_path}")
    
    return html_template

def markdown_file_to_html(
    markdown_path: str,
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    将 Markdown 文件转换为 HTML
    
    Args:
        markdown_path: Markdown 文件路径
        output_path: 输出HTML文件路径，默认与输入文件同目录同名
        **kwargs: 传递给 markdown_to_html 的其他参数
        
    Returns:
        str: 生成的HTML内容
    """
    markdown_path = Path(markdown_path)
    
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown文件不存在: {markdown_path}")
    
    # 读取 Markdown 内容
    with open(markdown_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # 默认输出路径
    if output_path is None:
        output_path = str(markdown_path.with_suffix('.html'))
    
    # 默认标题为文件名
    if 'title' not in kwargs:
        kwargs['title'] = markdown_path.stem
    
    # 默认图片基础路径为 Markdown 文件所在目录
    if 'image_base_path' not in kwargs:
        kwargs['image_base_path'] = str(markdown_path.parent)
    
    return markdown_to_html(markdown_content, output_path, **kwargs)
