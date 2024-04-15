import React from "react";
import Link from "next/link";

const Header = () => {
  return (
    <header className="py-3 px-4 border-b ">
      <div>
        <div className="w-full mx-auto px-4 6 flex h-16 items-center justify-between">
          <div className="flex items-left">
            <Link href="/" className="ml-4 lg:ml-0">
              <h1 className="text-2xl font-bold flex">
                InterpVis (2024):
                <span className="font-normal ml-3">Big Feature Graph </span>
              </h1>
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
