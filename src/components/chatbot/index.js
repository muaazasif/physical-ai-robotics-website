import React, { useState } from 'react';
import styles from './chatbot.module.css';

export default function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const askAI = async () => {
    if (!question.trim()) return;

    // Naya sawal shuru karne se pehle purana answer clear karein
    setAnswer("");
    setLoading(true);

    try {
      // 1. URL mein '/ask' lazmi check karein agar FastAPI mein route wahan hai
      const res = await fetch("https://rag-book-chatboting-production.up.railway.app/ask", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Accept": "application/json" 
        },
        body: JSON.stringify({ question: question })
      });

      // 2. Agar server error de (HTML bhej de)
      if (!res.ok) {
        const errorText = await res.text();
        console.error("Server Response:", errorText);
        throw new Error(`Server returned ${res.status}. Check Railway Logs!`);
      }

      const data = await res.json();
      
      // 3. Answer check karein
      if (data && data.answer) {
        setAnswer(data.answer);
      } else {
        setAnswer("Server se sahi response nahi mila.");
      }

    } catch (err) {
      console.error("Chatbot Error:", err);
      setAnswer("Bhai masla ho gaya: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.chatContainer}>
      {isOpen && (
        <div className={styles.chatWindow}>
          <div className={styles.chatHeader}>DocuSaur AI Chatbot</div>
          <div className={styles.chatBody}>
            {/* Answer display area */}
            {answer && (
              <div className={styles.answerBox}>
                <strong>AI:</strong> {answer}
              </div>
            )}
            
            {loading && (
              <div className={styles.loadingBox}>
                <span>🤖 Soch raha hoon...</span>
              </div>
            )}
            
            {!answer && !loading && (
              <p className={styles.placeholderText}>Humanoid Robotics ke bare mein kuch bhi puchein!</p>
            )}
          </div>
          
          <div className={styles.chatInputArea}>
            <input 
              value={question} 
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && askAI()} // Enter se bhi chalega
              placeholder="Sawal puchein..." 
            />
            <button onClick={askAI} disabled={loading}>
              {loading ? "..." : "Bhejo"}
            </button>
          </div>
        </div>
      )}
      <button className={styles.chatButton} onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? "✖" : "🤖 Ask Book"}
      </button>
    </div>
  );
}
