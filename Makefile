# 数值代数上机作业 Makefile
# 用法:
#   make        -- 编译所有报告
#   make 14     -- 仅编译第 14 次作业报告
#   make clean  -- 清理编译生成的辅助文件，仅保留 .tex 和 .pdf

REPORTS_DIR = reports
PDFLATEX = xelatex

# 自动发现所有报告文件 (report*.tex)
REPORT_TEXS := $(wildcard $(REPORTS_DIR)/report*.tex)
REPORT_PDFS := $(REPORT_TEXS:.tex=.pdf)

# 默认目标：编译所有报告
.PHONY: all clean
all: $(REPORT_PDFS)

# 模式规则：将 .tex 编译为 .pdf（编译两次以解决交叉引用）
$(REPORTS_DIR)/%.pdf: $(REPORTS_DIR)/%.tex $(REPORTS_DIR)/homework.sty
	cd $(REPORTS_DIR) && $(PDFLATEX) -interaction=nonstopmode $(notdir $<) >/dev/null 2>&1 || true
	cd $(REPORTS_DIR) && $(PDFLATEX) -interaction=nonstopmode $(notdir $<) >/dev/null 2>&1 || true

# 支持 make <num> 编译指定作业，例如 make 14
.PHONY: $(filter 8 10 11 12 14,$(MAKECMDGOALS))
8 10 11 12 14: %: $(REPORTS_DIR)/report%.pdf

# 清理：删除编译生成的辅助文件，仅保留 .tex 和 .pdf
clean:
	find $(REPORTS_DIR) -type f \
		! -name '*.tex' \
		! -name '*.pdf' \
		! -name '*.sty' \
		-delete
