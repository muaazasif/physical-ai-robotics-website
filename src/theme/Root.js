import React from 'react';
import Chatbot from '@site/src/components/Chatbot';

// The Root component wraps the entire application (Home, Docs, Blog, etc.)
export default function Root({children}) {
  return (
    <>
      {children}
      <Chatbot />
    </>
  );
}
