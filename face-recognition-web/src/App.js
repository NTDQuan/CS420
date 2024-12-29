import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import List from "./pages/List.jsx";
import Layout from "./components/shared/Layout.jsx";
import ListPerson from "./pages/ListPerson.jsx";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout/>}>
          <Route index element={<List/>}/>
        </Route>
        <Route path="/person" element={<Layout/>}>
          <Route index element={<ListPerson/>}/>
        </Route>
      </Routes>
    </Router>
  );
}

export default App;