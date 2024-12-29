import React from 'react'
import RecentAccess from '../components/RecentAccess'

const List = () => {
  return (
    <div className='flex flex-col gap-4'>
      <div className='flex flex-row gap-4 w-full'>
        <RecentAccess/>
      </div>
    </div>
  )
}

export default List
