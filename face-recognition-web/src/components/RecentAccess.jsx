import React, { useEffect, useState } from 'react';
import axios from 'axios';

const getStatusColor = (status) => {
    switch (status) {
      case 'Accept':
        return 'text-green-500';
      case 'Reject':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
};

const RecentAccess = () => {
  const [accessLogs, setAccessLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAccessLogs = async () => {
      try {
        const response = await axios.get('http://localhost:8000/access_logs');
        setAccessLogs(response.data.access_logs);
      } catch (error) {
        console.error("Error fetching access logs:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchAccessLogs();
  }, []);

  return (
    <div className="bg-white px-4 pt-3 pb-4 rounded-sm border border-gray-200 flex-1">
      <span className="text-gray-700 font-bold">Recent Access</span>
      <div className="border-x border-gray-200 rounded-sm mt-3">
        {loading ? (
          <div className="text-center py-4">Loading...</div>
        ) : (
          <table className="w-full text-gray-700 border-collapse">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="px-3 py-2 text-left">Id</th>
                <th className="px-3 py-2 text-left">Image</th>
                <th className="px-3 py-2 text-left">Personnel ID</th>
                <th className="px-3 py-2 text-left">Personnel Name</th>
                <th className="px-3 py-2 text-left">Access Date</th>
                <th className="px-3 py-2 text-left">Status</th>
              </tr>
            </thead>
            <tbody>
              {accessLogs.length > 0 ? (
                accessLogs.map((access) => (
                  <tr key={access.id} className="border-b border-gray-200">
                    <td className="px-3 py-2">{access.id}</td>
                    <td className="px-3 py-2">
                      <img
                        src={`http://localhost:8000/${access.image}`}
                        alt={`Personnel ${access.personnel_name}`}
                        className="w-10 h-10 object-cover rounded-full"
                      />
                    </td>
                    <td className="px-3 py-2">{access.person_id}</td>
                    <td className="px-3 py-2">{access.person_name}</td>
                    <td className="px-3 py-2">{access.access_date}</td>
                    <td className={`px-3 py-2 ${getStatusColor(access.status)}`}>{access.status}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="6" className="px-3 py-2 text-center text-gray-500">
                    No recent access data available.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
export default RecentAccess
