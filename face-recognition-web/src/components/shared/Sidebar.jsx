import React from 'react'
import { Link } from 'react-router-dom'

const linkClasses = 'flex items-center gap-2 font-light px-3 py-2 hover:bg-neutral-700 hover:no-underline active:bg-neutral-600 rounded-sm text-base'

const Sidebar = () => {
  return (
    <div className='bg-neutral-900 w-60 p-3 flex-col text-white'>
      <div className='flex items-center gap-2 px-1 py-3'>
        <span className='text-neutral-100 text-lg'>Admin</span>
      </div>
      <div className='flex-1 py-8 flex flex-col gap-0.5'>
        <SidebarLink name='Access list' path='/'/>
        <SidebarLink name='Person list' path='/person'/>
      </div>
    </div>
  )
}

function SidebarLink({ name, path }) { 
    return (
        <Link to={path} className={linkClasses}>
            {name}
        </Link>
    )
}

export default Sidebar
