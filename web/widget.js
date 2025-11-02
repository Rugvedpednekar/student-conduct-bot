// === Basic settings ===
const API_BASE = "http://127.0.0.1:8000"; // change if you deploy
const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("q");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const suggestEl = document.getElementById("suggest");

let typingRow = null;

// === Health check ===
(async function health(){
  try{
    const r = await fetch(`${API_BASE}/health`);
    if(r.ok){ statusEl.textContent = "Online"; statusEl.style.background = "rgba(255,255,255,.22)"; }
    else { statusEl.textContent = "API error"; }
  }catch{
    statusEl.textContent = "Offline";
  }
})();

// === Utilities ===
function el(tag, attrs={}, children=[]){
  const n = document.createElement(tag);
  Object.entries(attrs).forEach(([k,v])=>{ if(k==="class") n.className=v; else if(k==="html") n.innerHTML=v; else n.setAttribute(k,v);});
  (Array.isArray(children)?children:[children]).filter(Boolean).forEach(c=>n.appendChild(typeof c==="string"?document.createTextNode(c):c));
  return n;
}

function scrollToBottom(){ messagesEl.scrollTop = messagesEl.scrollHeight; }

function addMessage(text, role="bot", citations=[]){
  const avatar = el("div",{class:`avatar ${role}`}, role==="bot"?"AI":"You");
  const bubble = el("div",{class:"bubble", html: escapeAndFormat(text)});
  const row = el("div",{class:`msg ${role}`}, [avatar, bubble]);

  if(role==="bot" && citations?.length){
    const meta = el("div",{class:"meta"});
    citations.forEach(c=>{
      const chip = el("span",{class:"chip"}, `${c.source || "Student Handbook"}, p.${c.page}`);
      chip.title = c.url || "";
      if(c.url){ chip.onclick = ()=> window.open(c.url, "_blank"); chip.style.cursor="pointer"; }
      meta.appendChild(chip);
    });
    bubble.appendChild(meta);
  }

  messagesEl.appendChild(row);
  scrollToBottom();
}

function escapeAndFormat(text){
  // Simple sanitize + preserve line breaks and basic lists
  const esc = text.replace(/[&<>]/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[s]));
  return esc.replace(/\n/g,"<br>");
}

function showTyping(){
  const dots = el("div",{class:"typer"}, [el("span",{class:"dot"}), el("span",{class:"dot"}), el("span",{class:"dot"})]);
  typingRow = el("div",{class:"msg bot"}, [el("div",{class:"avatar bot"},"AI"), el("div",{class:"bubble"}, dots)]);
  messagesEl.appendChild(typingRow);
  scrollToBottom();
}
function hideTyping(){
  if(typingRow){ typingRow.remove(); typingRow=null; }
}

// === Ask flow ===
async function ask(q){
  if(!q.trim()) return;
  addMessage(q,"user");
  inputEl.value = "";
  hideTyping(); showTyping();

  try{
    const r = await fetch(`${API_BASE}/ask`, {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ question:q })
    });
    const data = await r.json();

    // Expecting { answer: str, citations: [{page, source, url}]? } from your API
    const text = data.answer || "Sorry, I couldn’t find that in the Student Handbook.";
    const cits = Array.isArray(data.citations) ? data.citations : parseCitationsFromText(text);
    hideTyping();
    addMessage(text,"bot",cits);
  }catch(e){
    hideTyping();
    addMessage("I couldn’t reach the server. Please try again.", "bot");
  }
}

function parseCitationsFromText(text){
  // Fallback: look for “(Student Handbook, p.X)”
  const re = /\(Student Handbook,\s*p\.(\d+)\)/gi;
  const out = [];
  let m;
  while((m = re.exec(text)) !== null){
    out.push({ page: m[1], source:"Student Handbook" });
  }
  return out;
}

// === Events ===
sendBtn.addEventListener("click", ()=> ask(inputEl.value));
inputEl.addEventListener("keydown", (e)=>{
  if(e.key==="Enter" && !e.shiftKey){
    e.preventDefault();
    ask(inputEl.value);
  }
});
suggestEl.addEventListener("click", (e)=>{
  const chip = e.target.closest(".chip");
  if(!chip) return;
  ask(chip.getAttribute("data-q"));
});

// Welcome + first hint
addMessage("Hi! I can answer questions about the UHart Student Handbook and cite the exact page. How can I help today?");
