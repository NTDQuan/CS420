import React, { useState, useEffect } from 'react';
import axios from 'axios';
import PersonnelList from '../components/PersonnelList';
import AddModal from '../components/AddModal';

const ListPerson = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [personList, setPersonList] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchPersonList = async () => {
    try {
      const response = await axios.get('http://localhost:8000/persons');
      setPersonList(response.data.persons);
    } catch (error) {
      console.error("Error fetching person list:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPersonList();
  }, []);

  return (
    <div className='flex flex-col gap-4'>
      <div>
        <button 
          onClick={() => setIsModalOpen(true)}
          className="bg-[#FAFBFC] p-3 rounded-lg cursor-pointer border-[1px] border-solid text-[rgba(27, 31, 35, 0.15)]"
        >
          Add
        </button>
      </div>
      <div className='flex flex-row gap-4 w-full'>
        <PersonnelList personList={personList} setPersonList={setPersonList} loading={loading} />
      </div>
      <AddModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} fetchPersonList={fetchPersonList} />
    </div>
  );
};

export default ListPerson;
