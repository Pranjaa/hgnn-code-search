import tkinter as tk
from tkinter import font, scrolledtext
from threading import Thread
import time
from inference import infer, code_map
from config import TOP_K

class CodeSearchUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Semantic Code Search")
        self.root.geometry("1000x800")
        self.root.configure(bg="#2b2d42") 

        self.heading_font = font.Font(family="Helvetica", size=18, weight="bold")
        self.query_font = font.Font(family="Helvetica", size=12)
        self.code_font = font.Font(family="Courier", size=11)

        self.results = []
        self.result_index = 0
        self.loading = False

        self.build_ui()

    def build_ui(self):
        tk.Label(self.root, text="Semantic Code Search", font=self.heading_font,
                 bg="#2b2d42", fg="#edf2f4").pack(pady=(30, 10))

        query_container = tk.Frame(self.root, bg="#2b2d42")
        query_container.pack(pady=(10, 20))

        tk.Label(query_container, text="Enter your query:", font=self.query_font,
                 bg="#2b2d42", fg="#edf2f4").pack(anchor="center", pady=(0, 8))

        self.query_box = tk.Text(query_container, height=2, width=60, font=self.query_font,
                                 wrap=tk.WORD, bd=2, relief="groove",
                                 bg="#8d99ae", fg="#1a1a1a", insertbackground="black")
        self.query_box.pack(padx=10)

        self.search_btn = tk.Button(self.root, text="Search", command=self.search_query,
                                    font=self.query_font, bg="#9a8c98", fg="white", padx=15, pady=6)
        self.search_btn.pack(pady=10)

        self.status_label = tk.Label(self.root, text="", font=self.query_font,
                                     bg="#2b2d42", fg="#edf2f4")
        self.status_label.pack(pady=5)

        self.code_container = tk.Frame(self.root, bg="#edf2f4", bd=2, relief="ridge")
        self.code_container.pack(padx=60, pady=20, fill="both", expand=True)

        self.code_scroll = scrolledtext.ScrolledText(self.code_container, font=self.code_font,
                                                     wrap=tk.WORD, bg="#f1f1f1", fg="#2b2d42",
                                                     insertbackground="black")
        self.code_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        self.code_scroll.config(state=tk.DISABLED)

        self.counter_label = tk.Label(self.root, text="", font=self.query_font,
                                      bg="#2b2d42", fg="#edf2f4")
        self.counter_label.pack(pady=(0, 5))

        self.prev_btn = tk.Button(self.root, text="⬅", font=self.query_font,
                                  bg="#8d99ae", command=self.prev_result)
        self.next_btn = tk.Button(self.root, text="➡", font=self.query_font,
                                  bg="#8d99ae", command=self.next_result)

        self.exit_btn = tk.Button(self.root, text="Exit", font=self.query_font,
                                  command=self.root.quit, bg="#d90429", fg="white")
        self.exit_btn.pack(pady=(0, 20))

    def search_query(self):
        query = self.query_box.get("1.0", tk.END).strip()
        if not query:
            self.status_label.config(text="⚠ Please enter a query.")
            return

        self.status_label.config(text="⏳ Searching")
        self.code_scroll.config(state=tk.NORMAL)
        self.code_scroll.delete("1.0", tk.END)
        self.code_scroll.config(state=tk.DISABLED)
        self.hide_nav_buttons()
        self.counter_label.config(text="")
        self.loading = True
        self.animate_loading()
        Thread(target=self.run_inference_thread, args=(query,), daemon=True).start()

    def animate_loading(self):
        def loop():
            dots = ""
            while self.loading:
                dots = (dots + ".") if len(dots) < 3 else ""
                self.status_label.config(text=f"⏳ Searching{dots}")
                time.sleep(0.5)
        Thread(target=loop, daemon=True).start()

    def run_inference_thread(self, query):
        self.results = infer(query, k=TOP_K)
        self.result_index = 0
        self.loading = False
        self.root.after(0, self.update_display)

    def update_display(self):
        if self.results:
            self.display_result()
            self.show_nav_buttons()
            self.status_label.config(text="Results loaded.")
        else:
            self.code_scroll.config(state=tk.NORMAL)
            self.code_scroll.delete("1.0", tk.END)
            self.code_scroll.insert(tk.END, "[No results found.]")
            self.code_scroll.config(state=tk.DISABLED)
            self.status_label.config(text="⚠ No matching results.")

    def display_result(self):
        idx, _ = self.results[self.result_index]
        code = code_map.get(str(idx), "[Code not found]")

        self.code_scroll.config(state=tk.NORMAL)
        self.code_scroll.delete("1.0", tk.END)
        self.code_scroll.insert(tk.END, code)
        self.code_scroll.config(state=tk.DISABLED)

        self.counter_label.config(text=f"Rank: {self.result_index + 1} / {len(self.results)}")

        self.prev_btn.config(state=tk.NORMAL if self.result_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.result_index < len(self.results) - 1 else tk.DISABLED)

    def show_nav_buttons(self):
        self.prev_btn.place(x=20, rely=1.0, anchor="sw")
        self.next_btn.place(x=self.root.winfo_width() - 20, rely=1.0, anchor="se")

    def hide_nav_buttons(self):
        self.prev_btn.place_forget()
        self.next_btn.place_forget()

    def prev_result(self):
        if self.result_index > 0:
            self.result_index -= 1
            self.display_result()

    def next_result(self):
        if self.result_index < len(self.results) - 1:
            self.result_index += 1
            self.display_result()

if __name__ == "__main__":
    root = tk.Tk()
    app = CodeSearchUI(root)
    root.mainloop()
