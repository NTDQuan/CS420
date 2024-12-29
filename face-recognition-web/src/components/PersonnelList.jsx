import React from 'react';
import axios from 'axios';

const PersonnelList = ({ personList, setPersonList, loading }) => {
  const handleDelete = async (personId) => {
    try {
      const response = await axios.post(
        'http://localhost:8000/delete_person',
        { id: String(personId) },  // Correct payload structure
        {
          headers: {
            'Content-Type': 'application/json',  // Ensure correct header
          },
        }
      );
      if (response.status === 200) {
        // Remove the deleted person from the state
        setPersonList(personList.filter((person) => person.id !== personId));
      }
    } catch (error) {
      console.error('Error deleting person:', error);
    }
  };

  return (
    <div className="bg-white px-4 pt-3 pb-4 rounded-sm border border-gray-200 flex-1">
      <span className="text-gray-700 font-bold">Personnel List</span>
      <div className="border-x border-gray-200 rounded-sm mt-3">
        <table className="w-full text-gray-700 border-collapse">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="px-3 py-2 text-left">Id</th>
              <th className="px-3 py-2 text-left">Image</th>
              <th className="px-3 py-2 text-left">Personnel Name</th>
              <th className="px-3 py-2 text-left">Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan="6" className="px-3 py-2 text-center text-gray-500">
                  Loading...
                </td>
              </tr>
            ) : personList.length > 0 ? (
              personList.map((personnel) => (
                <tr key={personnel.id} className="border-b border-gray-200">
                  <td className="px-3 py-2">{personnel.id}</td>
                  <td className="px-3 py-2">
                    <img
                      src={`http://localhost:8000/${personnel.image}`}
                      alt={`Personnel ${personnel.name}`}
                      className="w-10 h-10 object-cover rounded-full"
                    />
                  </td>
                  <td className="px-3 py-2">{personnel.name}</td>
                  <td className="px-3 py-2">
                    <button 
                      className="text-red-500 hover:text-red-700" 
                      onClick={() => handleDelete(personnel.id)}>
                      Delete
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="6" className="px-3 py-2 text-center text-gray-500">
                  No personnel data available.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PersonnelList;